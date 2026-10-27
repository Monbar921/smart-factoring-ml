"""score_delay: оценка задержки оплаты открытого УПД на дату t.

    from score_delay import score_delay, load_tables
    tables = load_tables("delivery/batch_01")                       # папка партии с CSV по контракту данных
    score_delay(["UPD-0000123", "UPD-0000124"], "2025-12-30", tables)

Вход: id УПД (строка или список) или таблица с колонкой upd_id; дата t; таблицы партии (путь или результат load_tables).
Выход: таблица с колонками upd_id, as_of_date, p_7, p_14, p_30, p_90, expected_days, model_version, status.
status: ok или no_history (нет закрытых УПД ни у пары, ни у покупателя на дату t; вероятности пусты).
Используются только УПД, подписанные не позже t, и платежи с датой не позже t. Оценивается открытый УПД, срок оплаты которого
не наступил (категории 1–4); для остальных выбрасывается ValueError."""
import datetime as dt
import os

import numpy as np
import pandas as pd

import dl_common as dc
import dl_model as dm

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("DELAY_MODEL_PATH", os.path.join(HERE, "delay_model_1_0.pkl"))
_model = None
_tables = {}


def get_model():
    global _model
    if _model is None:
        _model = dm.DelayModel.load(MODEL_PATH)
    return _model


def load_tables(path):
    if path not in _tables:
        _tables[path] = dc.load_csv_batch(path)
    return _tables[path]


def feature_rows(tab, ids, t_day, industries):
    """Признаки открытых УПД ids на дату t по таблицам, урезанным до t. Возвращает (U, rows, X)."""
    ids = list(ids)
    known = tab["upd"][tab["upd"].upd_id.isin(ids)]
    if len(known) != len(set(ids)):
        raise ValueError("УПД не найдены в таблицах: %s" % sorted(set(ids) - set(known.upd_id))[:5])
    late = known[known.upd_d > t_day]
    if len(late):
        raise ValueError("УПД подписаны после даты t: %s" % list(late.upd_id)[:5])
    U, A = dc.reconcile(tab, upto=t_day, buyers=set(known.buyer_inn))
    pos = {u: i for i, u in enumerate(U.upd_id.values)}
    idx = np.array([pos[u] for u in ids])
    bad = [u for u, i in zip(ids, idx) if not np.isnan(U.close_d.values[i]) or U.plan_d.values[i] < t_day]
    if bad:
        raise ValueError("УПД закрыты или просрочены на дату t (оцениваются категории 1–4): %s" % bad[:5])
    rows = pd.DataFrame(dict(u=idx, t=t_day, j=0))
    X = dc.features(tab, U, A, rows, industries)
    return U, rows, X


def score_delay(upd, t, tables, model=None):
    model = model or get_model()
    tab = load_tables(tables) if isinstance(tables, str) else tables
    if isinstance(upd, str):
        ids = [upd]
    elif isinstance(upd, pd.DataFrame):
        ids = list(upd.upd_id)
    else:
        ids = list(upd)
    t_ts = pd.Timestamp(t)
    t_day = int(dc.day_int(pd.Series([t_ts]))[0])
    U, rows, X = feature_rows(tab, ids, t_day, model.industries)
    nohist = (X.pair_n.values == 0) & (X.buy_n.values == 0)
    out = pd.DataFrame(dict(upd_id=ids, as_of_date=t_ts.strftime("%Y-%m-%d")))
    for c in ["p_7", "p_14", "p_30", "p_90", "expected_days"]:
        out[c] = np.nan
    if (~nohist).any():
        P = model.predict(X[~nohist])
        for c in P.columns:
            out.loc[~nohist, c] = P[c].values
    out["model_version"] = model.version
    out["status"] = np.where(nohist, "no_history", "ok")
    return out
