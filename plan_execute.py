"""Plan-Execute-Replan агент служби підтримки на LangGraph.

Запит клієнта спершу розбивається на явний план кроків, після чого
кроки виконуються по одному, а після кожного виконаного кроку окрема
LLM-ланка (replanner) вирішує, що робити далі:
  - `continue` — план ще актуальний, виконати наступний крок як є;
  - `replan`   — замінити залишок плану новими кроками (з'явились нові
                 дані, крок провалився, потрібне уточнення тощо);
  - `finish`   — інформації достатньо (або далі йти нікуди) — дати
                 клієнту фінальну відповідь.

Домен, tools, guard і logger — ті самі, що й у hw1 (`agent.py`,
`tools.py`, `safety.py`, `logger.py`), перевикористані напряму.
"""

import operator
from typing import Annotated, List, Literal, Optional, Tuple, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field, field_validator

from agent import SYSTEM_PROMPT, SupportResponse, llm, llm_structured, tools
from logger import TrajectoryLogger
from safety import RunGuard

# ── Structured outputs ───────────────────────────────────────────
class Plan(BaseModel):
    """Покроковий план виконання запиту клієнта."""
    steps: List[str] = Field(
        description=(
            'Кроки плану в порядку виконання. Кожен крок — самодостатнє '
            'завдання, яке можна виконати одним tool-викликом або одним '
            'міркуванням. Без зайвих кроків: якщо для відповіді достатньо '
            'одного tool-виклику, план має містити рівно один крок.'
        )
    )

    @field_validator('steps')
    @classmethod
    def steps_not_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError('План має містити щонайменше один крок')
        return v


class Act(BaseModel):
    """Рішення replanner-а щодо подальшого виконання плану."""
    action: Literal['continue', 'replan', 'finish'] = Field(description=(
        "'continue' — продовжити виконання поточного плану без змін; "
        "'replan' — замінити залишок плану новими кроками (поле steps обов'язкове); "
        "'finish' — завершити і дати фінальну відповідь клієнту (поле response обов'язкове)."
    ))
    steps: Optional[List[str]] = Field(default=None, description="Новий залишок плану, лише якщо action='replan'.")
    response: Optional[str] = Field(default=None, description="Фінальна відповідь клієнту, лише якщо action='finish'.")


# ── State ─────────────────────────────────────────────────────────
class PlanExecuteState(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple[str, str]], operator.add]
    response: Optional[str]
    final_response: Optional[SupportResponse]
    step_count: int
    # guard/logger — не reduced поля: створюються заново на кожен
    # app.invoke() і мутуються по посиланню (як і в agent.py).
    guard: RunGuard
    logger: TrajectoryLogger

# ── LLM-ланки ────────────────────────────────────────────────────
# method='function_calling': надійніше за строгий json_schema через
# проксі, як і в agent.py.
llm_planner = llm.with_structured_output(Plan, method='function_calling')
llm_replanner = llm.with_structured_output(Act, method='function_calling')

# ReAct-субагент, який виконує рівно один крок плану за виклик — той
# самий набір tools, що й у hw1.
step_executor = create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ('system', SYSTEM_PROMPT + (
        '\n\nПобудуй план виконання запиту клієнта у вигляді списку простих, '
        'самодостатніх кроків.'
    )),
    ('human', '{input}'),
])

REPLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ('system', SYSTEM_PROMPT),
    ('human', (
        'Запит клієнта: {input}\n\n'
        'Залишок плану:\n{plan}\n\n'
        'Уже виконані кроки та їх результати:\n{past_steps}\n\n'
        'Виріши, що робити далі:\n'
        "- якщо залишкові кроки плану досі актуальні — action='continue';\n"
        "- якщо потрібно скоригувати подальші кроки (нові дані, помилка tool, "
        "зайвий/відсутній крок) — action='replan' і подай оновлений залишок кроків "
        "(лише ті, що ще не виконані);\n"
        "- якщо зібраної інформації достатньо для відповіді клієнту, або залишкові "
        "кроки виконати неможливо (дані відсутні/невалідні), або залишок плану "
        "порожній — action='finish' і сформулюй response українською."
    )),
])


def _format_plan(steps: List[str]) -> str:
    return '\n'.join(f'{i + 1}. {s}' for i, s in enumerate(steps)) or '(порожньо)'


def _format_past_steps(past_steps: List[Tuple[str, str]]) -> str:
    if not past_steps:
        return '(ще немає виконаних кроків)'
    return '\n'.join(f'- {step} → {result}' for step, result in past_steps)

# ── Вузол planner ────────────────────────────────────────────────
def planner_node(state: PlanExecuteState) -> dict:
    """Побудувати початковий план виконання запиту клієнта."""
    traj_logger: TrajectoryLogger = state['logger']
    plan = llm_planner.invoke(PLANNER_PROMPT.invoke({'input': state['input']}))
    traj_logger.log('planner', {'steps': plan.steps})
    return {'plan': plan.steps, 'past_steps': []}

# ── Вузол executor ───────────────────────────────────────────────
def executor_node(state: PlanExecuteState) -> dict:
    """Виконати перший крок поточного плану (один крок плану за ітерацію)."""
    guard: RunGuard = state['guard']
    traj_logger: TrajectoryLogger = state['logger']
    step_count = state.get('step_count', 0)

    stop_reason = guard.check_step_limit(step_count) or guard.check_timeout()
    if stop_reason:
        traj_logger.log('executor_stopped', {'reason': stop_reason, 'step': step_count})
        return {
            'response': (
                f'⚠️ {stop_reason} Ось часткова відповідь на основі вже зібраної '
                f'інформації:\n{_format_past_steps(state["past_steps"])}'
            ),
            'step_count': step_count,
        }

    plan = state['plan']
    if not plan:
        traj_logger.log('executor_stopped', {'reason': 'Порожній план', 'step': step_count})
        return {'response': 'Немає кроків для виконання.', 'step_count': step_count}

    current_step = plan[0]
    task = (
        f'Повний план для запиту клієнта:\n{_format_plan(plan)}\n\n'
        f'Виконай зараз лише крок 1: {current_step}'
    )
    result = step_executor.invoke({'messages': [HumanMessage(content=task)]})
    step_result = result['messages'][-1].content

    traj_logger.log('executor', {
        'step': step_count + 1,
        'plan_step': current_step,
        'result_preview': str(step_result)[:300],
    })
    return {
        'past_steps': [(current_step, step_result)],
        'plan': plan[1:],
        'step_count': step_count + 1,
    }

# ── Вузол replanner ──────────────────────────────────────────────
def replanner_node(state: PlanExecuteState) -> dict:
    """Вирішити: продовжити план, замінити його чи завершити фінальною відповіддю."""
    traj_logger: TrajectoryLogger = state['logger']

    act = llm_replanner.invoke(REPLANNER_PROMPT.invoke({
        'input': state['input'],
        'plan': _format_plan(state['plan']),
        'past_steps': _format_past_steps(state['past_steps']),
    }))

    traj_logger.log('replanner', {
        'action': act.action,
        'new_steps': act.steps,
        'response_preview': (act.response or '')[:300],
    })

    if act.action == 'finish' or not state['plan']:
        return {'response': act.response or 'Не вдалося сформувати відповідь.'}
    if act.action == 'replan' and act.steps:
        return {'plan': act.steps}
    return {}  # 'continue' — план лишається без змін

# ── Вузол respond (structured output) ───────────────────────────
def respond_node(state: PlanExecuteState) -> dict:
    """Перетворити фінальну текстову відповідь на структурований Pydantic-об'єкт."""
    traj_logger: TrajectoryLogger = state['logger']
    structured = llm_structured.invoke([
        SystemMessage(content='Перетвори відповідь агента підтримки клієнтів у структурований формат.'),
        HumanMessage(content=state['response']),
    ])
    traj_logger.log('respond', {
        'final_text_preview': state['response'][:300],
        'resolved': structured.resolved,
    })
    return {'final_response': structured}

# ── Router-и ─────────────────────────────────────────────────────
def after_executor(state: PlanExecuteState) -> Literal['replanner', 'respond']:
    """Якщо guard зупинив виконання (max_steps/timeout) чи план порожній — одразу respond."""
    return 'respond' if state.get('response') else 'replanner'

def after_replanner(state: PlanExecuteState) -> Literal['executor', 'respond']:
    """Якщо replanner вирішив 'finish' (response вже сформовано) — respond, інакше далі виконувати план."""
    return 'respond' if state.get('response') else 'executor'

# ── Граф ─────────────────────────────────────────────────────────
graph = StateGraph(PlanExecuteState)
graph.add_node('planner', planner_node)
graph.add_node('executor', executor_node)
graph.add_node('replanner', replanner_node)
graph.add_node('respond', respond_node)

graph.add_edge(START, 'planner')
graph.add_edge('planner', 'executor')
graph.add_conditional_edges('executor', after_executor, {'replanner': 'replanner', 'respond': 'respond'})
graph.add_conditional_edges('replanner', after_replanner, {'executor': 'executor', 'respond': 'respond'})
graph.add_edge('respond', END)

app = graph.compile()

# ── Запуск ───────────────────────────────────────────────────────
if __name__ == '__main__':
    guard = RunGuard()
    traj_logger = TrajectoryLogger()

    query = 'Перевір статус замовлення У-0387776 в базі і заодно відстеж посилку — чи збігається інформація?'
    result = app.invoke({
        'input': query,
        'plan': [],
        'past_steps': [],
        'response': None,
        'final_response': None,
        'step_count': 0,
        'guard': guard,
        'logger': traj_logger,
    })

    traj_logger.save('trajectory_plan_execute.json')

    final: SupportResponse = result['final_response']
    print('Past steps:')
    for step, step_result in result['past_steps']:
        print(f'  - {step} → {str(step_result)[:120]}')
    print('\nStructured response:')
    print(final.model_dump_json(indent=2, exclude_none=True))
    print(f"Steps executed: {result['step_count']}")
    print('Trajectory saved: trajectory_plan_execute.json')
