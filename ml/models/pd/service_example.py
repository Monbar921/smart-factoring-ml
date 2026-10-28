"""Пример: как собрать вход модели PD из покупателя запроса /v1/pd/score и вернуть ответ в формате сервиса.
Это образец, а не готовый модуль: Denis встраивает логику в app.models.pd_trained.create().score(...).
Проверен на выдуманном покупателе: возвращает PD, уровень A-D и три причины."""
import datetime as dt
import numpy as np
import pandas as pd
from pd_model import PDModel

# строка формы -> поле запроса (рубли)
LINE = {"2110": "revenue", "2120": "cost_of_sales", "2200": "sales_profit", "2330": "interest_payable",
        "2400": "net_profit", "1200": "current_assets", "1230": "receivables", "1250": "cash", "1300": "equity",
        "1410": "long_term_borrowings", "1500": "short_term_liabilities", "1510": "short_term_borrowings",
        "1520": "payables", "1600": "assets_total"}
NO_VALUE = {"log_revenue", "log_assets", "okved_section", "region_group", "simplified_form", "no_prev_report",
            "receivables_faster_than_revenue", "payables_faster_than_revenue", "loss", "two_years_losses",
            "negative_equity", "revenue_drop_30", "is_moscow_mo"}


def rfsd_region(region):
    """Регион из запроса -> значение справочника RFSD. Модели важны три значения; остальное 'other'.
    Формат поля region в запросе подправь под реальный (здесь разбор по подстрокам)."""
    r = (region or "").lower()
    if r == "moscow reg." or "московск" in r or "подмосков" in r:
        return "moscow reg."
    if r == "moscow city" or "москва" in r:
        return "moscow city"
    if r == "sankt-petersburg" or "санкт" in r or "sankt" in r:
        return "sankt-petersburg"
    return "other"


def raw_row(buyer, calc_date):
    """Одна строка сырых данных для модели из покупателя запроса. None, если модель не определена (нужен базовый PD)."""
    fin = {f["year"]: f for f in buyer.get("financials") or []}
    if not fin:
        return None
    T = max(fin)
    if T < calc_date.year - 1:                                   # NO_RECENT_REPORT
        return None
    r = {"inn": buyer["inn"], "okved_section": buyer.get("okved_section"), "region": rfsd_region(buyer.get("region")),
         "simplified": int(bool(fin[T].get("simplified"))),
         "age": T - dt.date.fromisoformat(buyer["registration_date"]).year}
    for suffix, year in (("", T), ("_t1", T - 1), ("_t2", T - 2)):
        for code, name in LINE.items():
            v = (fin.get(year) or {}).get(name)
            r[f"line_{code}{suffix}"] = np.nan if v is None else v / 1000.0   # рубли -> тысячи рублей
    if not r["line_2110"] >= 120_000:                            # выручка ниже 120 млн ₽: LOW_REVENUE
        return None
    return r


def score(model, buyers, calc_date):
    rows = [raw_row(b, calc_date) for b in buyers]
    ok = [r for r in rows if r is not None]
    out = {}
    if ok:
        raw = pd.DataFrame(ok)
        X = model.features(raw)
        p, lvl, fac = model.predict_pd(X), model.level(model.predict_pd(X)), model.top_factors(X, 3)
        for r, pi, li, fi in zip(ok, p, lvl, fac):
            out[r["inn"]] = dict(pd=float(pi), level=str(li), reasons=[dict(
                feature=f["feature"], title=f["name_ru"][:1].upper() + f["name_ru"][1:],
                value=None if f["feature"] in NO_VALUE or f["value"] is None else float(f["value"]),
                direction="INCREASES_RISK" if f["direction"] == "raises_risk" else "DECREASES_RISK",
                contribution=f["shap"]) for f in fi])
    return out  # покупателей, которых нет в out, считать базовым PD (pd_source = BASE)
