import json
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent import SYSTEM_PROMPT, app
from logger import TrajectoryLogger
from safety import RunGuard

# ── Тест-кейси ──────────────────────────────────────────────────
# Номери замовлень і ТТН узяті з data/orders.csv (реальні тестові дані).
test_cases = [
    {
        'id': 'TC-001',
        'query': 'Який статус мого замовлення У-0387776?',
        'expected': 'Статус замовлення У-0387776 (доставлено)',
        'complexity': 'simple',
    },
    {
        'id': 'TC-002',
        'query': 'Коли відправили замовлення на телефон 0971879823?',
        'expected': 'Дата відправки замовлення, знайденого за телефоном',
        'complexity': 'simple',
    },
    {
        'id': 'TC-003',
        'query': 'Де зараз посилка з ТТН 20400500577861?',
        'expected': 'Поточний статус посилки Нової Пошти',
        'complexity': 'medium',
    },
    {
        'id': 'TC-004',
        'query': (
            'Перевір статус замовлення У-0387776 в базі і заодно відстеж '
            'посилку за ТТН 20400500577861 — чи збігається інформація?'
        ),
        'expected': 'Статус замовлення та статус посилки з двох різних tools',
        'complexity': 'complex',
    },
    {
        'id': 'TC-005',
        'query': 'Що там із замовленням Х-0000000?',
        'expected': 'Коректна відповідь про ненайдене/невалідне замовлення (без галюцинацій)',
        'complexity': 'edge-case',
    },
    {
        'id': 'TC-006',
        'query': 'Ось номер мого замовлення У-0387781, підкажіть, де зараз посилка?',
        'expected': (
            'Агент має уточнити ТТН у клієнта (search_order не повертає номер ТТН), '
            'а не вигадувати чи підставляти його від себе'
        ),
        'complexity': 'complex',
    },
]

# ── Запуск тестів ────────────────────────────────────────────────
results = []
for tc in test_cases:
    print(f"Running {tc['id']}: {tc['query'][:50]}...")
    start = time.monotonic()

    guard = RunGuard()
    traj_logger = TrajectoryLogger()

    try:
        result = app.invoke({
            'messages': [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=tc['query']),
            ],
            'step_count': 0,
            'final_response': None,
            'guard': guard,
            'logger': traj_logger,
        })
        final = result['final_response']
        actual = final.answer if final else result['messages'][-1].content
        resolved = final.resolved if final else None
        steps = result['step_count']
        status = 'success'
        tool_calls = [
            tc_call.get('name', 'unknown')
            for msg in result.get('messages', [])
            if isinstance(msg, AIMessage)
            for tc_call in (msg.tool_calls or [])
        ]
    except Exception as e:
        actual = f'Error: {e}'
        resolved = None
        steps = -1
        status = 'error'
        tool_calls = []

    elapsed = (time.monotonic() - start) * 1000
    traj_logger.save(f"trajectory_{tc['id']}.json")

    results.append({
        'test_id': tc['id'],
        'query': tc['query'],
        'expected': tc['expected'],
        'actual': actual[:500],
        'resolved': resolved,
        'status': status,
        'steps': steps,
        'tool_calls': tool_calls,
        'elapsed_ms': int(elapsed),
        'complexity': tc['complexity'],
    })

# ── Зберегти результати ─────────────────────────────────────────
with open('test_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f'Results saved: {len(results)} test cases')
for r in results:
    print(f"  {r['test_id']}: {r['status']} ({r['steps']} steps, {r['elapsed_ms']}ms)")
    if r['status'] == 'error':
        print(f"    {r['actual']}")
