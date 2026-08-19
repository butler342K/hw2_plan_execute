import os
from typing import Annotated, Literal, Optional, TypedDict
import operator

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, SecretStr

from knowledge import search_knowledge
from logger import TrajectoryLogger
from safety import RunGuard
from tools import get_order_ship_date, search_order, track_parcel, track_ukrposhta_parcel

load_dotenv()

SYSTEM_PROMPT = (
    'Ти — асистент служби підтримки інтернет-магазину. '
    'Допомагаєш клієнтам дізнатись статус замовлення чи посилки (Нова Пошта / Укрпошта), '
    'а також відповідаєш на загальні питання про доставку, оплату, повернення товару '
    'та інші правила магазину. '
    'Використовуй доступні tools, щоб знайти реальні дані — ніколи не вигадуй '
    'статус, дату чи номер, якого немає в результаті виклику tool. '
    'Для конкретного замовлення чи посилки клієнта користуйся search_order, '
    'get_order_ship_date, track_parcel або track_ukrposhta_parcel. '
    'Для загальних довідкових питань (правила, умови, причини затримок, повернення тощо) '
    'користуйся search_knowledge. Сам вирішуй, який tool (або кілька) потрібен для запиту.'
)

# ── Structured output ────────────────────────────────────────────
class SupportResponse(BaseModel):
    """Структурована фінальна відповідь агента підтримки."""
    answer: str = Field(description='Відповідь клієнту природною мовою')
    order_number: Optional[str] = Field(default=None, description='Номер замовлення, якщо йшлося про нього')
    tracking_number: Optional[str] = Field(default=None, description='ТТН (Нова Пошта) або ШКІ (Укрпошта), якщо йшлося про нього')
    status: Optional[str] = Field(default=None, description='Розпізнаний статус замовлення чи посилки')
    resolved: bool = Field(description='Чи вдалося знайти відповідь на запит клієнта за допомогою tools')

# ── State ────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    step_count: int
    final_response: Optional[SupportResponse]
    # guard/logger — звичайні (не reduced) поля: створюються заново на кожен
    # app.invoke() і мутуються по посиланню, тому не течуть між запусками.
    guard: RunGuard
    logger: TrajectoryLogger

# ── LLM з tools ──────────────────────────────────────────────────
# Для локальної Ollama йдемо через нативний ChatOllama: OpenAI-сумісний
# ендпоінт Ollama (/v1/chat/completions) ігнорує 'think: false' і qwen3
# все одно генерує довгий <think>-блок перед кожною відповіддю.
_base_url = os.getenv('OPENAI_BASE_URL', '')
_is_local_ollama = 'localhost:11434' in _base_url or '127.0.0.1:11434' in _base_url

if _is_local_ollama:
    llm = ChatOllama(
        model=os.getenv('LLM_MODEL', 'qwen3:latest'),
        base_url=_base_url.removesuffix('/v1'),
        temperature=0.1,
        reasoning=False,
    )
else:
    llm = ChatOpenAI(
        model=os.getenv('LLM_MODEL', 'deepseek-v4-flash'),
        base_url=_base_url,
        api_key=SecretStr(os.getenv('OPENAI_API_KEY', '')),
        temperature=0.1,
    )

tools = [search_order, get_order_ship_date, track_parcel, track_ukrposhta_parcel, search_knowledge]
llm_with_tools = llm.bind_tools(tools)
# method='function_calling': ця модель через zen-проксі ненадійно віддає
# строгий 'json_schema' response_format, натомість добре працює tool-calling.
llm_structured = llm.with_structured_output(SupportResponse, method='function_calling')

# ── Вузол agent ──────────────────────────────────────────────────
def agent_node(state: AgentState) -> dict:
    """LLM приймає рішення: викликати tool чи завершити діалог.

    Тут же застосовуються захисні механізми: max_steps, timeout і
    детекція зациклення (повторюваних tool calls) — усі три перевіряє
    один RunGuard, переданий через state.
    """
    guard: RunGuard = state['guard']
    traj_logger: TrajectoryLogger = state['logger']
    step = state.get('step_count', 0)

    stop_reason = guard.check_step_limit(step) or guard.check_timeout()
    if stop_reason:
        traj_logger.log('agent_stopped', {'reason': stop_reason, 'step': step})
        response = AIMessage(content=(
            f'⚠️ {stop_reason} Ось часткова відповідь на основі вже зібраної інформації.'
        ))
        return {'messages': [response], 'step_count': step}

    response = llm_with_tools.invoke(state['messages'])

    if isinstance(response, AIMessage) and response.tool_calls:
        loop_reason = guard.check_loop(response.tool_calls)
        if loop_reason:
            traj_logger.log('loop_detected', {'reason': loop_reason, 'step': step + 1})
            response = AIMessage(content=(
                f'⚠️ {loop_reason} Ось часткова відповідь на основі вже зібраної інформації.'
            ))

    traj_logger.log('agent', {
        'step': step + 1,
        'tool_calls': [tc['name'] for tc in (response.tool_calls or [])] if isinstance(response, AIMessage) else [],
        'content_preview': (response.content or '')[:300],
    })
    return {
        'messages': [response],
        'step_count': step + 1,
    }

# ── Вузол tools (з логуванням траєкторії) ───────────────────────────
_tool_node = ToolNode(tools)

def tools_node(state: AgentState) -> dict:
    """Виконати tool calls і залогувати кожен виклик та його результат."""
    traj_logger: TrajectoryLogger = state['logger']
    last_msg = state['messages'][-1]
    result = _tool_node.invoke(state)

    for call, tool_msg in zip(last_msg.tool_calls, result['messages']):
        traj_logger.log('tools', {
            'tool': call['name'],
            'args': call['args'],
            'output_preview': str(tool_msg.content)[:300],
        })
    return result

# ── Вузол respond (structured output) ──────────────────────────────
def respond_node(state: AgentState) -> dict:
    """Перетворити фінальну текстову відповідь LLM на структурований Pydantic-об'єкт."""
    traj_logger: TrajectoryLogger = state['logger']
    final_text = state['messages'][-1].content
    structured = llm_structured.invoke([
        SystemMessage(content='Перетвори відповідь агента підтримки клієнтів у структурований формат.'),
        HumanMessage(content=final_text),
    ])
    traj_logger.log('respond', {
        'final_text_preview': final_text[:300],
        'resolved': structured.resolved,
    })
    return {'final_response': structured}

# ── Router ───────────────────────────────────────────────────────
def should_continue(state: AgentState) -> Literal['tools', 'respond']:
    """Якщо LLM повернув tool_calls → tools, інакше → respond."""
    last_msg = state['messages'][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return 'tools'
    return 'respond'

# ── Граф ─────────────────────────────────────────────────────────
graph = StateGraph(AgentState)
graph.add_node('agent', agent_node)
graph.add_node('tools', tools_node)
graph.add_node('respond', respond_node)

graph.add_edge(START, 'agent')
graph.add_conditional_edges('agent', should_continue, {'tools': 'tools', 'respond': 'respond'})
graph.add_edge('tools', 'agent')  # після tools → назад до agent
graph.add_edge('respond', END)

app = graph.compile()

# ── Запуск ───────────────────────────────────────────────────────
if __name__ == '__main__':
    guard = RunGuard()
    traj_logger = TrajectoryLogger()

    result = app.invoke({
        'messages': [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content='Який статус мого замовлення У-0387776 і коли його відправили?'),
        ],
        'step_count': 0,
        'final_response': None,
        'guard': guard,
        'logger': traj_logger,
    })

    traj_logger.save('trajectory.json')

    final: SupportResponse = result['final_response']
    print('Structured response:')
    print(final.model_dump_json(indent=2, exclude_none=True))
    print(f"Steps: {result['step_count']}")
    print('Trajectory saved: trajectory.json')
