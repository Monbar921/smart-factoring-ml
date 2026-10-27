"""Сверка оплат по приложению Б.2 ТЗ. Платежи распределяются по УПД пары «клиент — покупатель».
Суммы — целые копейки, даты — целые дни. Платежи пары обрабатываются в порядке (дата, номер).

Правила:
 1. номер УПД в назначении -> этот УПД;
 2. иначе номер договора -> УПД договора, начиная с самого старого;
 3. иначе -> открытые УПД покупателя (пары), начиная с самого старого;
 4. остаток платежа -> следующий по старшинству открытый УПД пары;
 5. остаток УПД считается, полностью оплаченный закрывается;
 6. если открытых УПД нет, платёж нераспределён.
Открытый УПД на дату платежа: подписан не позже даты платежа и остаток больше нуля."""
import re

RE_UPD = re.compile(r"УПД\s*№\s*(\S+)\s+от\s+(\d{2})\.(\d{2})\.(\d{4})")
RE_CTR = re.compile(r"договору\s*№\s*(\S+)")


def parse_purpose(text):
    m = RE_UPD.search(text)
    if m:
        return "upd", m.group(1)
    m = RE_CTR.search(text)
    if m:
        return "contract", m.group(1)
    return "none", None


def allocate_payment(open_list, rem, num_to_idx, kind, key, amount):
    """open_list — индексы УПД пары по возрастанию старшинства, подписанные не позже даты платежа.
    rem — список остатков (копейки), изменяется. Возвращает (список (idx, сумма), нераспределённый остаток)."""
    alloc = []
    left = amount
    order = []
    if kind == "upd" and key in num_to_idx:
        first = num_to_idx[key]
        if rem[first] > 0 and first in open_list:
            order.append(first)
    for i in open_list:
        if rem[i] > 0 and (not order or i != order[0]):
            order.append(i)
    for i in order:
        if left <= 0:
            break
        a = min(left, rem[i])
        if a > 0:
            rem[i] -= a
            left -= a
            alloc.append((i, a))
    return alloc, left


def reconcile_pair(upd_dates, upd_amounts, upd_numbers, pays):
    """upd_*: списки по УПД пары (в порядке старшинства: дата, затем номер). pays: список
    (дата, порядок, сумма_коп, назначение). Возвращает (rem, close_day, unalloc, details),
    где close_day[i] — день последнего платежа, закрывшего УПД, или None."""
    n = len(upd_dates)
    rem = list(upd_amounts)
    close_day = [None] * n
    num_to_idx = {upd_numbers[i]: i for i in range(n)}
    unalloc = []
    details = []
    for d, order, amount, purpose in sorted(pays, key=lambda x: (x[0], x[1])):
        kind, key = parse_purpose(purpose)
        # открытые на дату платежа
        open_list = [i for i in range(n) if upd_dates[i] <= d and rem[i] > 0]
        alloc, left = allocate_payment(open_list, rem, num_to_idx, kind, key, amount)
        for i, a in alloc:
            if rem[i] == 0:
                close_day[i] = d
        if left > 0:
            unalloc.append((d, order, left))
        details.append((d, order, alloc, left))
    return rem, close_day, unalloc, details
