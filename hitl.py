"""Human-in-the-loop (HITL): підтвердження ризикової дії перед виконанням.

Ризикова дія — `notify_accountant` (надсилання повідомлення бухгалтеру про
повернення коштів клієнту). Перед її виконанням граф зупиняється через
`interrupt()`, показує оператору деталі дії (tool, параметри) і чекає на
рішення. Підтримано три сценарії відповіді:
  - approve — виконати дію як є;
  - reject  — відхилити, дію не виконувати;
  - edit    — виконати з відредагованими параметрами.

Non-ризикові tools (`search_order`, `get_order_ship_date`, `track_parcel`,
`track_ukrposhta_parcel`) беремо напряму з tools.py і виконуємо без
підтвердження.

Checkpointer (`SqliteSaver`) тут ОБОВ'ЯЗКОВИЙ: `interrupt()` зупиняє
виконання вузла всередині графа, і саме checkpointer зберігає стан на паузі
на диск, щоб `Command(resume=...)` міг згодом продовжити виконання —
за тим самим thread_id, навіть з іншого процесу.
"""

import operator
import sqlite3
import uuid
from pathlib import Path
from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from agent import llm
from tools import get_order_ship_date, search_order, track_parcel, track_ukrposhta_parcel

# ── Ризиковий tool: повідомлення бухгалтеру ──────────────────────
class NotifyAccountantInput(BaseModel):
    """Параметри повідомлення бухгалтеру про повернення коштів."""
    order_number: str = Field(description='Номер замовлення (наприклад, "У-0387776")')
    amount: float = Field(description='Сума до повернення клієнту, грн')
    reason: str = Field(description='Причина повернення коштів')

@tool(args_schema=NotifyAccountantInput)
def notify_accountant(order_number: str, amount: float, reason: str) -> str:
    """Надіслати бухгалтеру повідомлення про повернення коштів клієнту.

    РИЗИКОВА ДІЯ (рухає гроші компанії) — виконується лише після
    підтвердження оператором через interrupt(), див. confirm_and_execute.
    Поки що "відправка" — це друк у консоль.
    """
    message = (
        f'📧 [ДО БУХГАЛТЕРА] Повернути {amount:.2f} грн клієнту '
        f'за замовленням {order_number}. Причина: {reason}'
    )
    print(message)
    return f'Повідомлення бухгалтеру надіслано: {order_number}, {amount:.2f} грн ({reason})'

SAFE_TOOLS = [search_order, get_order_ship_date, track_parcel, track_ukrposhta_parcel]
ALL_TOOLS = [*SAFE_TOOLS, notify_accountant]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
RISKY_TOOLS = {'notify_accountant'}

llm_with_tools = llm.bind_tools(ALL_TOOLS)

# ── State ─────────────────────────────────────────────────────────
class PlanExecuteState(TypedDict):
    plan: list[str]
    current_step: int
    results: Annotated[list[str], operator.add]
    pending_tool_calls: list[dict]
    pending_text: Optional[str]

# ── Вузол propose: LLM обирає tool(и) для поточного кроку ─────────
# Винесений в окремий вузол від виконання, щоб при resume після
# interrupt() не викликати LLM повторно (нижче, у confirm_and_execute,
# лежить лише детермінований код + interrupt).
def propose_node(state: PlanExecuteState) -> dict:
    step_idx = state['current_step']
    step = state['plan'][step_idx]
    context = '\n'.join(state['results']) or '(ще немає результатів)'

    response = llm_with_tools.invoke(
        f'Виконай крок плану: {step}\n\n'
        f'Результати попередніх кроків:\n{context}\n\n'
        'Виклич відповідний tool з правильними параметрами.'
    )
    tool_calls = response.tool_calls if isinstance(response, AIMessage) else []
    return {
        'pending_tool_calls': tool_calls,
        'pending_text': response.content if not tool_calls else None,
    }

# ── Резолюція рішення оператора ────────────────────────────────────
def _resolve_decision(decision: object, original_args: dict) -> tuple[str, dict]:
    """approve → виконати як є; edit → виконати зі зміненими параметрами; reject → не виконувати."""
    if not isinstance(decision, dict):
        return 'reject', original_args
    action = decision.get('action', 'reject')
    if action == 'approve':
        return 'approve', original_args
    if action == 'edit':
        return 'edit', {**original_args, **decision.get('args', {})}
    return 'reject', original_args

# ── Вузол confirm_and_execute: interrupt перед ризиковими tools ──
def confirm_and_execute(state: PlanExecuteState) -> dict:
    step_idx = state['current_step']
    tool_calls = state['pending_tool_calls']

    if not tool_calls:
        result = state['pending_text'] or '(порожня відповідь)'
        return {
            'current_step': step_idx + 1,
            'results': [f'Крок {step_idx + 1}: {result}'],
            'pending_tool_calls': [],
        }

    step_outputs = []
    for tc in tool_calls:
        name, args = tc['name'], tc['args']

        if name not in RISKY_TOOLS:
            result = TOOLS_BY_NAME[name].invoke(args)
            step_outputs.append(f'{name}: {result}')
            continue

        # ── INTERRUPT: зупинити граф і чекати на рішення оператора ──
        decision = interrupt({
            'action': name,
            'args': args,
            'message': (
                f'⚠️ Підтвердіть ризикову дію перед виконанням:\n'
                f'Tool: {name}\n'
                f'Параметри: {args}'
            ),
        })

        kind, final_args = _resolve_decision(decision, args)
        if kind == 'reject':
            reason = decision.get('reason', 'без причини') if isinstance(decision, dict) else 'без причини'
            step_outputs.append(f'{name} — ВІДХИЛЕНО оператором ({reason})')
            continue

        result = TOOLS_BY_NAME[name].invoke(final_args)
        label = 'ЗМІНЕНО оператором і виконано' if kind == 'edit' else 'ПІДТВЕРДЖЕНО і виконано'
        step_outputs.append(f'{name} [{label}]: {result}')

    return {
        'current_step': step_idx + 1,
        'results': [f'Крок {step_idx + 1}: ' + '; '.join(step_outputs)],
        'pending_tool_calls': [],
    }

# ── Router ───────────────────────────────────────────────────────
def should_continue(state: PlanExecuteState) -> Literal['propose', '__end__']:
    return 'propose' if state['current_step'] < len(state['plan']) else END

# ── Граф ─────────────────────────────────────────────────────────
graph = StateGraph(PlanExecuteState)
graph.add_node('propose', propose_node)
graph.add_node('confirm_and_execute', confirm_and_execute)

graph.add_edge(START, 'propose')
graph.add_edge('propose', 'confirm_and_execute')
graph.add_conditional_edges('confirm_and_execute', should_continue, {'propose': 'propose', END: END})

# ── Checkpointer (ОБОВ'ЯЗКОВИЙ для interrupt()/Command(resume=...)) ──
CHECKPOINT_DB = Path(__file__).parent / 'hitl_state.db'
_conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
checkpointer = SqliteSaver(_conn)

app = graph.compile(checkpointer=checkpointer)

# ── Демонстрація трьох сценаріїв ────────────────────────────────────
DEFAULT_PLAN = [
    'Перевір статус замовлення У-0387776 через search_order.',
    'Повідом бухгалтера (notify_accountant) про повернення 450.00 грн '
    'клієнту за замовленням У-0387776, причина: брак товару.',
]


def _initial_state(plan: list[str]) -> dict:
    return {
        'plan': plan,
        'current_step': 0,
        'results': [],
        'pending_tool_calls': [],
        'pending_text': None,
    }


def _print_results(state: dict) -> None:
    print('\nРезультати виконання:')
    for r in state.get('results', []):
        print(f'  {r}')


def run_scenario(title: str, resume_payload: dict) -> None:
    """Прогнати план до interrupt на ризиковому tool, показати деталі дії
    оператору і продовжити виконання з переданим рішенням (approve/reject/edit)."""
    thread_id = str(uuid.uuid4())
    config = {'configurable': {'thread_id': thread_id}}
    print(f'\n{"=" * 70}\n{title} (thread_id={thread_id})\n{"=" * 70}')

    result = app.invoke(_initial_state(DEFAULT_PLAN), config)

    interrupts = result.get('__interrupt__')
    if not interrupts:
        print('Граф завершився без interrupt (ризиковий tool не викликано).')
        _print_results(result)
        return

    payload = interrupts[0].value
    print(f"\n⏸  ГРАФ ЗУПИНЕНО перед ризиковою дією:\n{payload['message']}")
    print(f'\n▶ Рішення оператора: {resume_payload}')

    result = app.invoke(Command(resume=resume_payload), config)
    _print_results(result)


if __name__ == '__main__':
    run_scenario('Сценарій 1: APPROVE (підтвердити й виконати як є)', {'action': 'approve'})
    run_scenario(
        'Сценарій 2: REJECT (відхилити)',
        {'action': 'reject', 'reason': 'потребує додаткової перевірки документів'},
    )
    run_scenario(
        'Сценарій 3: EDIT (змінити параметри й виконати)',
        {'action': 'edit', 'args': {'amount': 300.0, 'reason': 'часткове повернення — товар підлягає обміну'}},
    )
