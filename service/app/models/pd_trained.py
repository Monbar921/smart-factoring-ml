"""Обученная модель PD покупателя коллеги (pd-1.0, app/models/pd): PD_MODEL_MODULE=app.models.pd_trained.

Модель считается на отчётности, которую присылает backend (Р-20), а не по ИНН из хранилища RFSD, как score_inn.py
коллеги: строки форм года отчётности T = год до даты расчёта и двух предыдущих, суммы - в тыс. руб., как при обучении.

Где модель не определена, - базовый PD эвристики по разделу ОКВЭД с пометкой «мало данных» (п. 4.2):
нет отчётности за прошлый год, выручка ниже 120 млн, компания моложе года (features.low_data_reason)
и разделы ОКВЭД K и O, которых не было в обучении.
"""

import math
import os
from datetime import date

import lightgbm as lgb
import numpy as np
import pandas as pd

from .. import features
from ..schemas import Buyer, PdResult, Reason
from . import _pickle
from .heuristic import BASE_PD_BY_SECTION, BASE_PD_DEFAULT, HeuristicPdModel
from .pd.pd_features import LINES
from .pd.pd_model import PDModel

MODEL_PATH = os.environ.get("PD_MODEL_PATH", os.path.join(os.path.dirname(__file__), "pd", "pd_model.pkl"))
EXCLUDED_SECTIONS = {"K", "O"}  # отфильтрованы при обучении (07_extract_scoring_store.py)
THOUSAND = 1000.0

# строка формы -> поле Financials запроса
LINE_FIELDS = {
    "2110": "revenue", "1200": "current_assets", "1250": "cash", "1500": "short_term_liabilities", "1300": "equity",
    "1600": "assets_total", "1410": "long_term_borrowings", "1510": "short_term_borrowings", "2400": "net_profit",
    "2200": "sales_profit", "2330": "interest_payable", "1230": "receivables", "1520": "payables", "2120": "cost_of_sales",
}

# регион RFSD (латиницей); модель различает только Москву, область и Санкт-Петербург
REGIONS = {
    "москва": "moscow city", "г. москва": "moscow city", "город москва": "moscow city",
    "московская область": "moscow reg.", "московская обл.": "moscow reg.",
    "санкт-петербург": "sankt-petersburg", "г. санкт-петербург": "sankt-petersburg",
}


def rfsd_region(region: str | None) -> str | None:
    if region is None:
        return None
    value = region.strip().lower()
    return REGIONS.get(value, value)


class TrainedPdModel:
    stub = False

    def __init__(self, model: PDModel):
        self.model = model
        self.version = model.meta.get("version", "pd-1.0") if isinstance(model.meta, dict) else "pd-1.0"
        trained = model.meta.get("trained_at") if isinstance(model.meta, dict) else None
        self.trained_at = date.fromisoformat(str(trained)[:10]) if trained else None
        self.base = HeuristicPdModel()

    def score(self, buyers: list[Buyer], calc_date: date) -> list[PdResult]:
        results: dict[str, PdResult] = {}
        scored = []
        for buyer in buyers:
            reason = features.low_data_reason(buyer, calc_date)
            if reason is not None:
                results[buyer.inn] = self.base.score([buyer], calc_date)[0]
            elif (buyer.okved_section or "").upper() in EXCLUDED_SECTIONS:
                results[buyer.inn] = PdResult(
                    inn=buyer.inn, status="SCORED", pd=BASE_PD_BY_SECTION.get(buyer.okved_section.upper(), BASE_PD_DEFAULT),
                    pd_source="BASE", low_data=True, low_data_reason="EXCLUDED_INDUSTRY")
            else:
                scored.append(buyer)
        if scored:
            raw = pd.DataFrame([row(buyer, calc_date.year - 1) for buyer in scored])
            frame = self.model.features(raw)
            pds = self.model.predict_pd(frame)
            factors = self.model.top_factors(frame, 3)
            for buyer, value, top in zip(scored, pds, factors):
                results[buyer.inn] = PdResult(inn=buyer.inn, status="SCORED", pd=round(float(value), 6), pd_source="MODEL",
                                              reasons=[reason_of(factor) for factor in top])
        return [results[buyer.inn] for buyer in buyers]


def row(buyer: Buyer, year: int) -> dict:
    """Строка компания-год, как в хранилище RFSD: строки T, T-1 (_t1), T-2 (_t2) в тыс. руб."""
    reports = {item.year: item for item in buyer.financials}
    out = {"inn": buyer.inn, "okved_section": buyer.okved_section, "region": rfsd_region(buyer.region)}
    for suffix, report in (("", reports.get(year)), ("_t1", reports.get(year - 1)), ("_t2", reports.get(year - 2))):
        for line in LINES:
            value = getattr(report, LINE_FIELDS[line]) if report is not None else None
            out[f"line_{line}{suffix}"] = np.nan if value is None else value / THOUSAND
    current = reports.get(year)
    cost = current.cost_of_sales if current is not None else None
    out["line_2120"] = np.nan if cost is None else cost / THOUSAND
    out["simplified"] = int(bool(current.simplified)) if current is not None else 0
    # полных лет на конец года отчётности
    out["age"] = float(year - buyer.registration_date.year) if buyer.registration_date else np.nan
    return out


def reason_of(factor: dict) -> Reason:
    value = factor["value"]
    numeric = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
    name = factor["name_ru"]
    return Reason(
        feature=factor["feature"],
        title=name[:1].upper() + name[1:],
        value=round(float(value), 4) if numeric and not math.isnan(float(value)) else None,
        direction="INCREASES_RISK" if factor["direction"] == "raises_risk" else "DECREASES_RISK",
        contribution=round(abs(float(factor["shap"])), 4),
    )


def create() -> TrainedPdModel:
    state = _pickle.load(MODEL_PATH, {"pd_features": "app.models.pd.pd_features", "pd_model": "app.models.pd.pd_model"})
    return TrainedPdModel(PDModel(state["fb"], lgb.Booster(model_str=state["booster_txt"]), state["calibrator"], state["cuts"],
                                  state["meta"]))
