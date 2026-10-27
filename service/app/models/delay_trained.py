"""Обученная модель задержки оплаты коллеги (delay-1.0, app/models/delay): DELAY_MODEL_MODULE=app.models.delay_trained.

Переходник JSON -> таблицы модели (README модели, «Подключение к сервису»):
- УПД и их закрытия берутся из history, как их прислал backend (сверку оплат модель заново не делает);
- оплаченная часть оцениваемого УПД на дату расчёта = amount - outstanding;
- покупатель: PD из запроса, уровень A-D по порогам модели 0,4% / 1,5% / 6% (в backend свои пороги уровней -
  они для решения и карточки, модель обучена на своих); коэффициенты - по последней отчётности из buyers;
- отрасль клиента - client_industry (коды backend переводятся в значения обучения).

Результат: p_delay = p_<порог>, p_overdue_90 = p_90, expected_delay_days = expected_days; нет закрытых УПД ни у пары,
ни у покупателя - NO_PAYMENT_HISTORY; закрытый или просроченный УПД - ERROR (модель для него не определена).
"""

import os
from datetime import date

import numpy as np
import pandas as pd

from ..schemas import Buyer, DelayRequest, DelayResult
from . import _pickle
from .delay import dl_common as dc
from .delay.dl_model import DelayModel
from .pd_trained import rfsd_region

MODEL_PATH = os.environ.get("DELAY_MODEL_PATH", os.path.join(os.path.dirname(__file__), "delay", "delay_model_1_0.pkl"))
LEVEL_CUTS = (0.004, 0.015, 0.06)  # пороги уровней модели PD коллеги: A < 0,4% <= B < 1,5% <= C < 6% <= D
INDUSTRIES = {
    "CONSTRUCTION": "строительство",
    "PROFESSIONAL": "профессиональные_услуги",
    "MANUFACTURING": "производство",
    "TRADE": "торговля",
}
ID_WIDTH = 20


def level(pd_value: float | None) -> str | None:
    if pd_value is None or np.isnan(pd_value):
        return None
    return "A" if pd_value < LEVEL_CUTS[0] else "B" if pd_value < LEVEL_CUTS[1] else "C" if pd_value < LEVEL_CUTS[2] else "D"


def sort_id(value: str) -> str:
    """Порядок УПД одного дня - по id, как при обучении; числовые id backend дополняются нулями, чтобы 99 < 100."""
    return value.zfill(ID_WIDTH) if value.isdigit() else value


class TrainedDelayModel:
    stub = False
    trained_at = None

    def __init__(self, model: DelayModel):
        self.model = model
        self.version = model.version

    def score(self, request: DelayRequest) -> list[DelayResult]:
        t_day = int(dc.day_int(pd.Series([pd.Timestamp(request.calc_date)]))[0])
        upds, clients = upd_table(request, t_day)
        buyers = buyer_table(request)
        frame = score_frame(self.model, upds, clients, buyers,
                            {upd.upd_id: (upd.amount - upd.outstanding) for upd in request.upds}, t_day)
        results = []
        for upd in request.upds:
            row = frame.loc[upd.upd_id]
            if row.status == "error":
                results.append(DelayResult(upd_id=upd.upd_id, status="ERROR", error=row.error))
            elif row.status == "no_history":
                results.append(DelayResult(upd_id=upd.upd_id, status="NO_PAYMENT_HISTORY", no_pair_history=True))
            else:
                results.append(DelayResult(
                    upd_id=upd.upd_id, status="SCORED",
                    p_delay=round(float(row[f"p_{request.delay_threshold_days}"]), 6),
                    p_overdue_90=round(float(row.p_90), 6),
                    expected_delay_days=round(float(row.expected_days), 2),
                    no_pair_history=bool(row.no_pair_history)))
        return results


def upd_table(request: DelayRequest, t_day: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """УПД (история и оцениваемые) в колонках dl_common.normalize + close_d; клиенты с отраслью."""
    rows, seen = [], set()
    for i, item in enumerate(request.history):
        upd_id = item.upd_id or f"h{i:08d}"
        seen.add(upd_id)
        closed = item.status == "PAID" and item.paid_date is not None and item.paid_date <= request.calc_date
        rows.append((upd_id, item.number or "", item.client_inn, item.buyer_inn, item.upd_date, item.planned_due_date,
                     item.amount, item.paid_date if closed else None))
    for upd in request.upds:
        if upd.upd_id not in seen:
            rows.append((upd.upd_id, upd.number or "", upd.client_inn, upd.buyer_inn, upd.upd_date, upd.planned_due_date,
                         upd.amount, None))
    frame = pd.DataFrame(rows, columns=["upd_id", "upd_number", "client_id", "buyer_inn", "upd_date", "plan_date", "amount",
                                        "close_date"])
    upds = pd.DataFrame(dict(
        upd_id=frame.upd_id.astype(str).values, upd_number=frame.upd_number.astype(str).values,
        client_id=frame.client_id.astype(str).values, buyer_inn=frame.buyer_inn.astype(str).values,
        upd_d=dc.day_int(frame.upd_date), plan_d=dc.day_int(frame.plan_date),
        amount_k=np.rint(frame.amount.astype(float).values * 100).astype(np.int64),
        close_d=np.array([np.nan if value is None else float(dc.day_int(pd.Series([value]))[0]) for value in frame.close_date])))
    upds = upds[upds.upd_d <= t_day]
    industries = {}
    for upd in request.upds:
        if upd.client_industry:
            industries[upd.client_inn] = INDUSTRIES.get(upd.client_industry.upper(), upd.client_industry)
    clients = pd.DataFrame(dict(client_id=list(industries), industry=list(industries.values())), dtype=object)
    return upds, clients


def buyer_table(request: DelayRequest) -> pd.DataFrame:
    """Покупатель на дату расчёта: PD и уровень, коэффициенты последней отчётности (README модели, «Соответствие входа»)."""
    pds = {upd.buyer_inn: upd.buyer_pd for upd in request.upds}
    rows = []
    for buyer in request.buyers:
        rows.append(buyer_row(buyer, pds.get(buyer.inn)))
    known = {row["buyer_inn"] for row in rows}
    rows += [buyer_row(Buyer(inn=inn), value) for inn, value in pds.items() if inn not in known]
    return pd.DataFrame(rows)


def buyer_row(buyer: Buyer, pd_value: float | None) -> dict:
    last = max(buyer.financials, key=lambda item: item.year) if buyer.financials else None
    ratio = lambda a, b: np.nan if a is None or not b else a / b  # noqa: E731
    payables_days = ratio(last.payables, abs(last.cost_of_sales) if last and last.cost_of_sales else None) if last else np.nan
    return dict(
        buyer_inn=buyer.inn, okved_section=buyer.okved_section, region=rfsd_region(buyer.region),
        age_years=float(last.year - buyer.registration_date.year) if last and buyer.registration_date else np.nan,
        revenue_mln=last.revenue / 1e6 if last and last.revenue else np.nan,
        abs_liquidity=ratio(last.cash, last.short_term_liabilities) if last else np.nan,
        current_liquidity=ratio(last.current_assets, last.short_term_liabilities) if last else np.nan,
        payables_days=payables_days * 365 if not np.isnan(payables_days) else np.nan,
        pd=np.nan if pd_value is None else float(pd_value), risk_level=level(pd_value))


def score_frame(model: DelayModel, upds: pd.DataFrame, clients: pd.DataFrame, buyers: pd.DataFrame, paid: dict[str, float],
                t_day: int) -> pd.DataFrame:
    """Ядро переходника - и для запроса сервиса, и для сверки с delay_scores.csv: таблицы как у модели, дата t.

    upds - УПД с close_d (день закрытия или NaN), paid - оплаченная на t часть оцениваемых УПД (по id), руб.
    Возвращает по id оцениваемого УПД: status (ok / no_history / error), p_7…p_90, expected_days, no_pair_history.
    """
    upds = upds.assign(_key=upds.upd_id.map(sort_id))
    U = upds.sort_values(["client_id", "buyer_inn", "upd_d", "_key"], kind="stable").reset_index(drop=True)
    U["pair"] = pd.factorize((U.client_id + "|" + U.buyer_inn).values)[0]
    position = {upd_id: i for i, upd_id in enumerate(U.upd_id.values)}
    ids = list(paid)
    out = pd.DataFrame(index=pd.Index(ids, name="upd_id"),
                       columns=["status", "error", "p_7", "p_14", "p_30", "p_90", "expected_days", "no_pair_history"])
    valid = []
    for upd_id in ids:
        i = position.get(upd_id)
        if i is None:
            out.loc[upd_id, ["status", "error"]] = ["error", "УПД подписан после даты расчёта"]
        elif not np.isnan(U.close_d.values[i]) or U.plan_d.values[i] < t_day:
            out.loc[upd_id, ["status", "error"]] = ["error", "модель задержки определена только для открытых УПД до срока оплаты"]
        else:
            valid.append(upd_id)
    if not valid:
        return out
    ui = np.array([position[upd_id] for upd_id in valid])
    A = pd.DataFrame(dict(u=ui, d=t_day, a=np.rint([max(0.0, paid[upd_id]) * 100 for upd_id in valid]).astype(np.int64)))
    A = A[A.a > 0].sort_values(["u", "d"], kind="stable").reset_index(drop=True)
    tab = dict(by=buyers, cl=clients.astype(str) if len(clients) else pd.DataFrame(dict(client_id=[], industry=[])))
    X = dc.features(tab, U.drop(columns="_key"), A, pd.DataFrame(dict(u=ui, t=t_day, j=0)), model.industries)
    X.index = valid
    no_history = (X.pair_n.values == 0) & (X.buy_n.values == 0)
    out.loc[valid, "status"] = np.where(no_history, "no_history", "ok")
    out.loc[valid, "no_pair_history"] = (X.pair_n.values == 0)
    if (~no_history).any():
        P = model.predict(X[~no_history])
        for column in P.columns:
            out.loc[P.index, column] = P[column].values
    return out


def create() -> TrainedDelayModel:
    state = _pickle.load(MODEL_PATH, {"dl_model": "app.models.delay.dl_model"})
    model = DelayModel(state["features"], state["categorical"], state["industries"], state["boosters"], state["calibrators"],
                       state["reg"], state["meta"])
    model.version = state["version"]
    return TrainedDelayModel(model)
