"""Эвристические модели по умолчанию: работают, пока нет обученных моделей (Р-19).

Коэффициенты подобраны вручную для демо, не обучены. Заменяются моделями из ТЗ на PD и на задержку
через PD_MODEL_MODULE / DELAY_MODEL_MODULE (см. registry.py).
"""

import math
from datetime import date

from ..features import buyer_items, discipline, low_data_reason, pair_items, pd_features
from ..schemas import Buyer, DelayRequest, DelayResult, PdResult, Reason

# базовый PD по разделу ОКВЭД (п. 4.2: «мало данных»), условные значения
BASE_PD_BY_SECTION = {"F": 0.05, "C": 0.03, "G": 0.03, "H": 0.035, "M": 0.03}
BASE_PD_DEFAULT = 0.035


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def clip(value: float | None, low: float, high: float, default: float = 0.0) -> float:
    return default if value is None else max(low, min(high, value))


class HeuristicPdModel:
    version = "pd-heuristic-1"
    trained_at = None
    stub = True

    # (признак, подпись, вес, функция значения): вклад = вес × значение
    TERMS = [
        ("loss", "Убыток за год", 1.0, lambda f: 1.0 if f.loss else 0.0),
        ("loss_two_years", "Убыток два года подряд", 0.8, lambda f: 1.0 if f.loss_two_years else 0.0),
        ("negative_equity", "Отрицательный капитал", 1.8, lambda f: 1.0 if f.negative_equity else 0.0),
        ("equity_ratio", "Доля собственного капитала", -2.0, lambda f: clip(f.equity_ratio, -1, 1)),
        ("debt_to_revenue", "Займы к выручке", 0.7, lambda f: clip(f.debt_to_revenue, 0, 3)),
        ("current_ratio", "Текущая ликвидность", -0.9, lambda f: clip(f.current_ratio, 0, 3, 1)),
        ("abs_liquidity", "Абсолютная ликвидность", -1.5, lambda f: clip(f.abs_liquidity, 0, 1)),
        ("revenue_drop_30", "Падение выручки больше 30%", 1.0, lambda f: 1.0 if f.revenue_drop_30 else 0.0),
        ("net_margin", "Маржа чистой прибыли", -4.0, lambda f: clip(f.net_margin, -0.5, 0.5)),
    ]
    INTERCEPT = -3.2

    def score(self, buyers: list[Buyer], calc_date: date) -> list[PdResult]:
        return [self._score(buyer, calc_date) for buyer in buyers]

    def _score(self, buyer: Buyer, calc_date: date) -> PdResult:
        features = pd_features(buyer, calc_date)
        contributions = []
        for feature, title, weight, value_of in self.TERMS:
            value = value_of(features)
            contributions.append((feature, title, weight * value, getattr(features, feature)))
        reasons = [
            Reason(
                feature=feature,
                title=title,
                value=round(float(raw), 4) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else None,
                direction="INCREASES_RISK" if contribution > 0 else "DECREASES_RISK",
                contribution=round(abs(contribution), 4),
            )
            for feature, title, contribution, raw in sorted(contributions, key=lambda item: -abs(item[2]))[:3]
            if contribution != 0
        ]
        reason = low_data_reason(buyer, calc_date)
        if reason is not None:
            base = BASE_PD_BY_SECTION.get(buyer.okved_section or "", BASE_PD_DEFAULT)
            return PdResult(inn=buyer.inn, status="SCORED", pd=base, pd_source="BASE", low_data=True, low_data_reason=reason,
                            reasons=reasons)
        logit = self.INTERCEPT + sum(item[2] for item in contributions)
        pd = round(min(0.5, max(0.001, sigmoid(logit))), 6)
        return PdResult(inn=buyer.inn, status="SCORED", pd=pd, pd_source="MODEL", reasons=reasons)


class HeuristicDelayModel:
    version = "delay-heuristic-1"
    trained_at = None
    stub = True

    def score(self, request: DelayRequest) -> list[DelayResult]:
        return [self._score(request, upd) for upd in request.upds]

    def _score(self, request: DelayRequest, upd) -> DelayResult:
        buyer = discipline(buyer_items(request.history, upd), request.calc_date, request.delay_threshold_days)
        if buyer.paid == 0:
            # п. 4.5, ситуация 3: нет ни одного оплаченного УПД - модель не применяется
            return DelayResult(upd_id=upd.upd_id, status="NO_PAYMENT_HISTORY", no_pair_history=True)
        pair = discipline(pair_items(request.history, upd), request.calc_date, request.delay_threshold_days)
        no_pair_history = pair.paid == 0
        # п. 4.5, ситуация 4: нет истории пары - по дисциплине покупателя
        source = buyer if no_pair_history else pair
        size = clip(upd.amount / pair.typical_amount - 1 if pair.typical_amount else 0, 0, 3)
        logit = (-3.2 + 0.06 * min(source.avg_delay, 60) + 2.0 * source.share_late + 1.0 * buyer.share_open_overdue
                 + 6.0 * upd.buyer_pd + 0.3 * size + 0.004 * upd.deferral_days)
        logit_90 = -5.5 + 0.03 * min(buyer.avg_delay, 90) + 3.0 * buyer.share_open_overdue_30 + 10.0 * upd.buyer_pd
        expected = source.avg_delay * (1 + 0.5 * source.share_late)
        return DelayResult(
            upd_id=upd.upd_id,
            status="SCORED",
            p_delay=round(sigmoid(logit), 6),
            p_overdue_90=round(sigmoid(logit_90), 6),
            expected_delay_days=round(expected, 1),
            no_pair_history=no_pair_history,
        )
