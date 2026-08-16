import csv
import os

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

load_dotenv()

ORDERS_CSV_PATH = os.path.join(os.path.dirname(__file__), 'data', 'orders.csv')

NOVA_POSHTA_API_URL = 'https://api.novaposhta.ua/v2.0/json/'
NOVA_POSHTA_API_KEY = os.getenv('NOVA_POSHTA_API_KEY', '')

UKRPOSHTA_API_URL = 'https://www.ukrposhta.ua/status-tracking/0.1.0'
UKRPOSHTA_API_TOKEN = os.getenv('UKRPOSHTA_API_TOKEN', '')

STATUS_LABELS = {
    'Доставлен': 'доставлено',
    'Отменен': 'скасовано',
    'Отказ от покупки': 'відмова від покупки',
}

ORDER_NUMBER_PREFIXES = ('У-', 'М-', 'К-')

# ── Pydantic-схема для пошуку замовлення ─────────────────────────
class OrderSearchInput(BaseModel):
    """Параметри для пошуку замовлення."""
    query: str = Field(description='Номер замовлення (наприклад, "У-0387776") або телефон клієнта')

    @field_validator('query')
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v or len(v.strip()) < 3:
            raise ValueError('Запит для пошуку має містити мінімум 3 символи')
        v = v.strip()
        if v[0].isalpha() and not v.startswith(ORDER_NUMBER_PREFIXES):
            raise ValueError(
                f'Номер замовлення має починатися з {", ".join(ORDER_NUMBER_PREFIXES)}'
            )
        return v

def _load_orders() -> list[dict]:
    """Зчитати всі замовлення з data/orders.csv у список словників."""
    orders = []
    with open(ORDERS_CSV_PATH, encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f, delimiter=';', quotechar='"')
        for row in reader:
            if len(row) < 11 or not row[0]:
                continue
            orders.append({
                'order_number': row[0].strip(),
                'created_at': row[2].strip(),
                'status': row[3].strip(),
                'ship_date': row[4].split(' ')[0].strip(),
                'amount': row[5].strip(),
                'ttn': row[6].strip(),
                'customer_name': row[9].strip(),
                'customer_phone': row[10].strip(),
            })
    return orders

def _find_orders(query: str) -> list[dict]:
    """Знайти замовлення за номером (повністю або частково) чи телефоном клієнта."""
    orders = _load_orders()
    query_digits = ''.join(ch for ch in query if ch.isdigit())

    matches = [o for o in orders if query.lower() in o['order_number'].lower()]
    if not matches and query_digits:
        matches = [o for o in orders if query_digits in o['customer_phone']]
    return matches

# ── Tool з Pydantic-схемою ───────────────────────────────────────
@tool(args_schema=OrderSearchInput)
def search_order(query: str) -> str:
    """Знайти замовлення в базі за номером або телефоном клієнта.

    Використовуйте цей інструмент, коли потрібно дізнатися статус
    замовлення, дату відправки чи номер ТТН/ШКІ для подальшого
    відстеження посилки через track_parcel / track_ukrposhta_parcel.

    Args:
        query: Номер замовлення (повністю або частково) або телефон клієнта.

    Returns:
        Рядок зі статусом, датою відправки і ТТН (якщо є) знайдених замовлень.
    """
    matches = _find_orders(query)
    if not matches:
        return f'Замовлення за запитом "{query}" не знайдено.'

    lines = [
        f'{o["order_number"]} — статус: {STATUS_LABELS.get(o["status"], o["status"])}, '
        f'дата відправки: {o["ship_date"]}, '
        f'ТТН: {o["ttn"] or "немає"}, клієнт: {o["customer_name"]}'
        for o in matches[:5]
    ]
    if len(matches) > 5:
        lines.append(f'...і ще {len(matches) - 5} замовлень.')
    return '\n'.join(lines)

# ── Tool з Pydantic-схемою ───────────────────────────────────────
@tool(args_schema=OrderSearchInput)
def get_order_ship_date(query: str) -> str:
    """Отримати дату відправки замовлення за номером або телефоном клієнта.

    Використовуйте цей інструмент, коли потрібно дізнатися лише дату
    відправки замовлення. Ця дата вже міститься у вхідній таблиці
    замовлень (data/orders.csv), тому звернення до зовнішнього API
    не потрібне.

    Args:
        query: Номер замовлення (повністю або частково) або телефон клієнта.

    Returns:
        Рядок з датою відправки знайдених замовлень.
    """
    matches = _find_orders(query)
    if not matches:
        return f'Замовлення за запитом "{query}" не знайдено.'

    lines = [
        f'{o["order_number"]} — дата відправки: {o["ship_date"]}, '
        f'клієнт: {o["customer_name"]}'
        for o in matches[:5]
    ]
    if len(matches) > 5:
        lines.append(f'...і ще {len(matches) - 5} замовлень.')
    return '\n'.join(lines)

# ── Pydantic-схема для відстеження ТТН Нової Пошти ────────────────
class TrackParcelInput(BaseModel):
    """Параметри для відстеження посилки Нової Пошти."""
    ttn: str = Field(description='Номер ТТН (наприклад, "20450012345678")')

    @field_validator('ttn')
    @classmethod
    def ttn_valid(cls, v: str) -> str:
        v = ''.join(ch for ch in v if ch.isdigit())
        if len(v) < 12:
            raise ValueError('Номер ТТН має містити щонайменше 12 цифр')
        return v

# ── Tool з Pydantic-схемою ───────────────────────────────────────
@tool(args_schema=TrackParcelInput)
def track_parcel(ttn: str) -> str:
    """Відстежити посилку Нової Пошти за номером ТТН через офіційне API.

    Використовуйте цей інструмент, коли потрібно дізнатися
    поточний статус, місцезнаходження чи дату прибуття посилки.

    Args:
        ttn: Номер ТТН (накладної) Нової Пошти.

    Returns:
        Рядок зі статусом посилки та додатковою інформацією.
    """
    if not NOVA_POSHTA_API_KEY:
        return 'Помилка: не задано NOVA_POSHTA_API_KEY у змінних середовища.'

    payload = {
        'apiKey': NOVA_POSHTA_API_KEY,
        'modelName': 'TrackingDocument',
        'calledMethod': 'getStatusDocuments',
        'methodProperties': {
            'Documents': [{'DocumentNumber': ttn}],
        },
    }

    try:
        response = requests.post(NOVA_POSHTA_API_URL, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        return f'Помилка звернення до API Нової Пошти: {e}'

    if not data.get('success'):
        errors = data.get('errors') or data.get('warnings') or ['невідома помилка']
        return f'Не вдалося отримати статус ТТН {ttn}: {", ".join(errors)}'

    results = data.get('data') or []
    if not results:
        return f'ТТН {ttn} не знайдено.'

    info = results[0]
    lines = [
        f'ТТН {ttn} — статус: {info.get("Status", "невідомо")}',
        f'Звідки: {info.get("CitySender", "?")}, куди: {info.get("CityRecipient", "?")}',
    ]
    scheduled = info.get('ScheduledDeliveryDate')
    if scheduled:
        lines.append(f'Очікувана дата доставки: {scheduled}')
    actual = info.get('ActualDeliveryDate')
    if actual:
        lines.append(f'Фактична дата доставки: {actual}')
    warehouse = info.get('WarehouseRecipient')
    if warehouse:
        lines.append(f'Відділення отримувача: {warehouse}')
    return '\n'.join(lines)

# ── Pydantic-схема для відстеження ШКІ Укрпошти ───────────────────
class TrackUkrposhtaInput(BaseModel):
    """Параметри для відстеження відправлення Укрпошти."""
    barcode: str = Field(description='Номер поштового відправлення, ШКІ (наприклад, "0500100031143")')

    @field_validator('barcode')
    @classmethod
    def barcode_not_empty(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 8:
            raise ValueError('Номер ШКІ має містити щонайменше 8 символів')
        return v

# ── Tool з Pydantic-схемою ───────────────────────────────────────
@tool(args_schema=TrackUkrposhtaInput)
def track_ukrposhta_parcel(barcode: str) -> str:
    """Відстежити відправлення Укрпошти за ШКІ через офіційне API.

    Використовуйте цей інструмент, коли потрібно дізнатися
    поточний статус чи місцезнаходження відправлення Укрпошти.

    Args:
        barcode: Номер поштового відправлення (ШКІ) Укрпошти.

    Returns:
        Рядок зі статусом відправлення та додатковою інформацією.
    """
    if not UKRPOSHTA_API_TOKEN:
        return 'Помилка: не задано UKRPOSHTA_API_TOKEN у змінних середовища.'

    headers = {'Authorization': f'Bearer {UKRPOSHTA_API_TOKEN}'}
    params = {'barcode': barcode, 'lang': 'UA'}

    try:
        response = requests.get(
            f'{UKRPOSHTA_API_URL}/statuses/last', headers=headers, params=params, timeout=10
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        return f'Помилка звернення до API Укрпошти: {e}'

    if not data or not data.get('barcode'):
        return f'Відправлення {barcode} не знайдено.'

    lines = [
        f'Відправлення {barcode} — статус: {data.get("eventName", "невідомо")}',
        f'Відділення: {data.get("name", "?")} (індекс {data.get("index", "?")})',
        f'Дата події: {data.get("date", "?")}',
    ]
    reason = data.get('eventReason')
    if reason:
        lines.append(f'Деталі: {reason}')
    return '\n'.join(lines)

# ── Unit test ────────────────────────────────────────────────────
if __name__ == '__main__':
    # Валідація Pydantic працює:
    print('Testing Pydantic validation...')
    print('Query: У-')
    try:
        OrderSearchInput(query='У-')  # Повинно впасти (< 3 символів)
    except Exception as e:
        print(f'Validation error (expected): {e}')

    print('Query: X-0387776')
    try:
        OrderSearchInput(query='X-0387776')  # Повинно впасти (невірний префікс)
    except Exception as e:
        print(f'Validation error (expected): {e}')

    # Tool працює:
    print('Query: У-0387776')
    result = search_order.invoke({'query': 'У-0387776'})
    print(f'Order: {result}')

    print('Query: У-0387776')
    result = get_order_ship_date.invoke({'query': 'У-0387776'})
    print(f'Ship date: {result}')

    print('TTN: 123')
    try:
        TrackParcelInput(ttn='123')  # Повинно впасти (< 12 цифр)
    except Exception as e:
        print(f'Validation error (expected): {e}')

    print('TTN: 20400541439999')
    TrackParcelInput(ttn='20400541439999')  # Не повинно впасти
    print('Validation OK')

    print('TTN: 20400541439999')
    result = track_parcel.invoke({'ttn': '20400541439999'})
    print(f'Parcel: {result}')

    print('Barcode: 123')
    try:
        TrackUkrposhtaInput(barcode='123')  # Повинно впасти (< 8 символів)
    except Exception as e:
        print(f'Validation error (expected): {e}')

    print('Barcode: 0500100031143')
    TrackUkrposhtaInput(barcode='0505676508680')  # Не повинно впасти
    print('Validation OK')

    print('Barcode: 0505676508680')
    result = track_ukrposhta_parcel.invoke({'barcode': '0505676508680'})
    print(f'Parcel: {result}')

