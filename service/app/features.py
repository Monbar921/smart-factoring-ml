"""Признаки из сырых данных (Р-22): считаются здесь, тем же кодом при обучении и в сервисе.

PD: драйверы групп 1-4 п. 4.2 BRD по отчётности ГИР БО.
Задержка: платёжная дисциплина покупателя и пары по истории УПД (п. 4.3).
"""

from dataclasses import dataclass
from datetime import date

from .schemas import Buyer, DelayUpd, Financials, HistoryUpd

REVENUE_THRESHOLD = 120_000_000  # порог выручки модели PD, ниже - базовый PD (Приложение А)


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def report(buyer: Buyer, year: int) -> Financials | None:
    return next((item for item in buyer.financials if item.year == year), None)


def low_data_reason(buyer: Buyer, calc_date: date) -> str | None:
    """Базовый PD вместо модели (п. 4.2): нет отчётности за прошлый год, выручка ниже порога, компания моложе года.

    Причина из источника данных (Buyer.low_data_reason) важнее вычисленной: в партиях ML-команды отчётности нет совсем.
    """
    if buyer.low_data_reason is not None:
        return buyer.low_data_reason
    last = report(buyer, calc_date.year - 1)
    if last is None:
        return "NO_RECENT_REPORT"
    if last.revenue is None or last.revenue < REVENUE_THRESHOLD:
        return "LOW_REVENUE"
    if buyer.registration_date is not None and (calc_date - buyer.registration_date).days < 365:
        return "YOUNG_COMPANY"
    return None


@dataclass
class PdFeatures:
    """Драйверы п. 4.2 (номера - как в BRD)."""

    net_margin: float | None  # 1
    loss: bool  # 2
    loss_two_years: bool  # 3
    negative_equity: bool  # 4
    equity_ratio: float | None  # 5
    debt_to_revenue: float | None  # 6
    interest_coverage: float | None  # 7
    current_ratio: float | None  # 8
    abs_liquidity: float | None  # 9
    receivable_days: float | None  # 10
    payable_days: float | None  # 11
    revenue_change: float | None  # 12
    revenue_drop_30: bool  # 12
    simplified: bool  # 22
    age_years: float | None  # 17


def pd_features(buyer: Buyer, calc_date: date) -> PdFeatures:
    last = report(buyer, calc_date.year - 1) or (max(buyer.financials, key=lambda item: item.year) if buyer.financials else None)
    previous = report(buyer, last.year - 1) if last else None
    if last is None:
        return PdFeatures(None, False, False, False, None, None, None, None, None, None, None, None, False, False, None)
    borrowings = (last.long_term_borrowings or 0) + (last.short_term_borrowings or 0)
    revenue_change = ratio((last.revenue or 0) - (previous.revenue or 0), previous.revenue) if previous else None
    age = (calc_date - buyer.registration_date).days / 365 if buyer.registration_date else None
    return PdFeatures(
        net_margin=ratio(last.net_profit, last.revenue),
        loss=(last.net_profit or 0) < 0,
        loss_two_years=(last.net_profit or 0) < 0 and previous is not None and (previous.net_profit or 0) < 0,
        negative_equity=(last.equity or 0) < 0,
        equity_ratio=ratio(last.equity, last.assets_total),
        debt_to_revenue=ratio(borrowings, last.revenue),
        interest_coverage=ratio(last.sales_profit, last.interest_payable),
        current_ratio=ratio(last.current_assets, last.short_term_liabilities),
        abs_liquidity=ratio(last.cash, last.short_term_liabilities),
        receivable_days=None if ratio(last.receivables, last.revenue) is None else ratio(last.receivables, last.revenue) * 365,
        payable_days=None if ratio(last.payables, last.cost_of_sales) is None else ratio(last.payables, last.cost_of_sales) * 365,
        revenue_change=revenue_change,
        revenue_drop_30=revenue_change is not None and revenue_change < -0.3,
        simplified=bool(last.simplified),
        age_years=age,
    )


@dataclass
class Discipline:
    """Платёжная дисциплина по набору УПД на дату расчёта."""

    paid: int
    avg_delay: float  # средняя задержка оплаченных, дней (досрочная оплата - 0)
    share_late: float  # доля оплаченных позже порога задержки
    share_open_overdue: float  # доля открытых УПД с наступившим сроком
    share_open_overdue_30: float  # доля открытых с просрочкой больше 30 дней
    typical_amount: float | None  # медиана суммы УПД


def discipline(items: list[HistoryUpd], calc_date: date, threshold_days: int) -> Discipline:
    paid = [item for item in items if item.status == "PAID" and item.paid_date is not None]
    delays = [max(0, (item.paid_date - item.planned_due_date).days) for item in paid]
    opened = [item for item in items if item.status == "OPEN"]
    overdue = [item for item in opened if item.planned_due_date < calc_date]
    overdue_30 = [item for item in opened if (calc_date - item.planned_due_date).days > 30]
    amounts = sorted(item.amount for item in items)
    return Discipline(
        paid=len(paid),
        avg_delay=sum(delays) / len(delays) if delays else 0.0,
        share_late=sum(1 for delay in delays if delay > threshold_days) / len(delays) if delays else 0.0,
        share_open_overdue=len(overdue) / len(opened) if opened else 0.0,
        share_open_overdue_30=len(overdue_30) / len(opened) if opened else 0.0,
        typical_amount=amounts[len(amounts) // 2] if amounts else None,
    )


def pair_items(history: list[HistoryUpd], upd: DelayUpd) -> list[HistoryUpd]:
    return [item for item in history if item.buyer_inn == upd.buyer_inn and item.client_inn == upd.client_inn]


def buyer_items(history: list[HistoryUpd], upd: DelayUpd) -> list[HistoryUpd]:
    return [item for item in history if item.buyer_inn == upd.buyer_inn]
