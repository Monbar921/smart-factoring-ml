"""
score(inn): PD of a buyer, risk level A-D and the three main factors.

    from score_inn import score
    score("7707083893")
    -> {"inn": ..., "status": "ok", "financial_year": 2024, "cutoff_note": ..., "pd": 0.0123, "level": "B",
        "factors": [{"name": "маржа чистой прибыли", "direction": "raises_risk", "value": 0.0004, "text": "..."}, ...]}

Status values
    ok            - PD calculated from the latest reporting year found in the store
    no_financials - INN is not in the store: no filed report for that year, revenue below 120 mln RUB,
                    ОКВЭД sections K/O, or outlier. The model is not defined for such companies;
                    the product must treat them as 'no PD' (manual review), not as low risk.

The model is trained on cohort 2022 (report year 2022, cutoff 01.04.2023). Scoring a later report year
applies the same model to the same features; the PSI check for 2024 is in the report.
"""
import os
import numpy as np
import pandas as pd
from pd_model import PDModel

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("PD_MODEL_PATH", os.path.join(HERE, "pd_model.pkl"))
STORE_PATHS = os.environ.get("PD_STORE_PATHS", os.path.join(HERE, "scoring_store_2024.parquet")).split(os.pathsep)

_model, _store = None, None


def _load():
    global _model, _store
    if _model is None:
        _model = PDModel.load(MODEL_PATH)
    if _store is None:
        parts = [pd.read_parquet(p) for p in STORE_PATHS if os.path.exists(p)]
        if not parts:
            raise FileNotFoundError("scoring store not found: " + ", ".join(STORE_PATHS))
        _store = pd.concat(parts, ignore_index=True).sort_values(["inn", "financial_year"])
    return _model, _store


def _fmt(name_feature, v):
    if v is None:
        return "нет данных"
    if isinstance(v, str):
        return v
    if name_feature in ("age",):
        return f"{v:.0f} г."
    if name_feature.startswith(("log_",)):
        return f"{np.exp(v):,.0f} тыс. ₽".replace(",", " ")
    if name_feature in ("receivables_days", "payables_days"):
        return f"{v:.0f} дн."
    if name_feature in ("net_margin", "revenue_growth_1y", "revenue_growth_2y", "equity_change_1y", "equity_change_2y",
                        "equity_ratio", "debt_to_revenue"):
        return f"{v*100:.1f}%"
    if float(v) in (0.0, 1.0) and name_feature in __import__("pd_features").FLAG_COLS:
        return "да" if v else "нет"
    return f"{v:.2f}"


def score(inn: str, year: int | None = None, k: int = 3) -> dict:
    """PD, level and top-k factors for one INN. `year` = reporting year (default: latest in the store)."""
    model, store = _load()
    inn = str(inn).strip()
    rows = store[store.inn == inn]
    if year is not None:
        rows = rows[rows.financial_year == year]
    if rows.empty:
        return {"inn": inn, "status": "no_financials", "pd": None, "level": None, "factors": []}
    raw = rows.iloc[[-1]]
    X = model.features(raw)
    pdv = float(model.predict_pd(X)[0])
    lvl = str(model.level([pdv])[0])
    facs = model.top_factors(X, k)[0]
    for f in facs:
        f["text"] = f"{f['name_ru']}: {_fmt(f['feature'], f['value'])} — " + \
                    ("повышает риск" if f["direction"] == "raises_risk" else "снижает риск")
    return {"inn": inn, "status": "ok", "financial_year": int(raw.financial_year.iloc[0]), "pd": pdv, "level": lvl,
            "factors": [{"name": f["name_ru"], "feature": f["feature"], "direction": f["direction"], "value": f["value"], "text": f["text"]} for f in facs]}


def score_many(inns, year=None) -> pd.DataFrame:
    """Vectorised scoring for a list of INNs (PD, level, three factors as text)."""
    model, store = _load()
    s = store[store.inn.isin(set(map(str, inns)))]
    if year is not None:
        s = s[s.financial_year == year]
    s = s.groupby("inn").tail(1)
    return model.score_frame(s)
