"""Модель задержки оплаты по УПД: таблицы партии, сверка оплат, выборка, метки и признаки на дату t.

Один и тот же код считает признаки и для обучения, и для score_delay. Все признаки на дату t используют только
УПД, подписанные не позже t, и платежи с датой не позже t (проверяется отдельным тестом с урезанными таблицами).
Дни — целые числа (дней от 1970-01-01), суммы — целые копейки."""
import bisect
import os

import numpy as np
import pandas as pd

from alloc import reconcile_pair

BIG = 1 << 17
KS = (7, 14, 30, 90)
HIST_T = (0, 7, 14, 30, 90)          # пороги задержки, по которым считаем доли в истории
EPOCH = np.datetime64("1970-01-01", "D")
OKVED = list("ABCDEFGHIJKLMNOPQRSTU")
LEVELS = {"A": 0, "B": 1, "C": 2, "D": 3}
MOSCOW = {"moscow city", "moscow reg."}


def day_int(s):
    return (pd.to_datetime(s).values.astype("datetime64[D]") - EPOCH).astype(np.int64)


def int_day(d):
    return pd.Timestamp(EPOCH + np.timedelta64(int(d), "D"))


# ------------------------------------------------------------------ таблицы
def normalize(upd, pay, cl, by):
    """Приводит таблицы партии (файлы контракта или parquet) к одному виду."""
    u = pd.DataFrame(dict(
        upd_id=upd.upd_id.astype(str).values, upd_number=upd.upd_number.astype(str).values,
        client_id=upd.client_id.astype(str).values, buyer_inn=upd.buyer_inn.astype(str).values,
        upd_d=day_int(upd.upd_date), plan_d=day_int(upd.plan_pay_date),
        amount_k=np.rint(upd.amount_rub.astype(float).values * 100).astype(np.int64)))
    p = pd.DataFrame(dict(
        payment_id=pay.payment_id.astype(str).values, client_id=pay.client_id.astype(str).values,
        buyer_inn=pay.buyer_inn.astype(str).values, pay_d=day_int(pay.pay_date),
        amount_k=np.rint(pay.amount_rub.astype(float).values * 100).astype(np.int64),
        purpose=pay.purpose.astype(str).values)).sort_values("payment_id").reset_index(drop=True)
    p["order"] = np.arange(len(p))
    c = cl[["client_id", "industry"]].astype(str).reset_index(drop=True)
    b = by[["buyer_inn", "okved_section", "region", "age_years", "revenue_mln", "abs_liquidity", "current_liquidity",
            "payables_days", "pd", "risk_level"]].copy()
    b["buyer_inn"] = b.buyer_inn.astype(str)
    for col in ["age_years", "revenue_mln", "abs_liquidity", "current_liquidity", "payables_days", "pd"]:
        b[col] = pd.to_numeric(b[col], errors="coerce")
    return dict(upd=u.sort_values("upd_id").reset_index(drop=True), pay=p, cl=c, by=b.reset_index(drop=True))


def load_csv_batch(path):
    rd = lambda n: pd.read_csv(os.path.join(path, n + ".csv"), dtype=str, keep_default_na=False, na_values=[""],
                               encoding="utf-8")
    return normalize(rd("upd"), rd("payments"), rd("clients"), rd("buyers"))


# ------------------------------------------------------------------ сверка оплат (Б.2) с журналом распределения
def reconcile(tab, upto=None, buyers=None):
    """Возвращает (U, A). U — УПД в порядке пар и старшинства, колонки pair, close_d (NaN, если открыт на конец данных).
    A — события распределения: индекс строки U, день платежа, сумма в копейках. Если задано upto, берутся только
    УПД и платежи не позже этого дня; buyers — подмножество покупателей (для быстрого расчёта одного УПД)."""
    u, p = tab["upd"], tab["pay"]
    if upto is not None:
        u = u[u.upd_d <= upto]
        p = p[p.pay_d <= upto]
    if buyers is not None:
        buyers = set(buyers)
        u = u[u.buyer_inn.isin(buyers)]
        p = p[p.buyer_inn.isin(buyers)]
    u = u.sort_values(["client_id", "buyer_inn", "upd_d", "upd_id"], kind="stable").reset_index(drop=True)
    key = (u.client_id + "|" + u.buyer_inn).values
    code, uniq = pd.factorize(key)
    u["pair"] = code
    n = len(u)
    starts = np.flatnonzero(np.r_[True, code[1:] != code[:-1]]) if n else np.array([], dtype=int)
    ends = np.r_[starts[1:], n] if n else np.array([], dtype=int)
    pcode = pd.Series((p.client_id + "|" + p.buyer_inn).values).map({k: i for i, k in enumerate(uniq)})
    pg = {}
    for i, (d, o, a, pu, c) in enumerate(zip(p.pay_d.values, p.order.values, p.amount_k.values, p.purpose.values, pcode.values)):
        if c == c:
            pg.setdefault(int(c), []).append((int(d), int(o), int(a), pu))
    ud, ua, un = u.upd_d.tolist(), u.amount_k.tolist(), u.upd_number.tolist()
    close = np.full(n, np.nan)
    Au, Ad, Aa = [], [], []
    for gi, (s, e) in enumerate(zip(starts, ends)):
        rem, cd, _, det = reconcile_pair(ud[s:e], ua[s:e], un[s:e], pg.get(gi, []))
        for i, c in enumerate(cd):
            if c is not None:
                close[s + i] = c
        for d, o, alloc, left in det:
            for i, a in alloc:
                Au.append(s + i)
                Ad.append(d)
                Aa.append(a)
    u["close_d"] = close
    A = pd.DataFrame(dict(u=Au, d=Ad, a=Aa)).sort_values(["u", "d"], kind="stable").reset_index(drop=True)
    return u, A


# ------------------------------------------------------------------ строки выборки и метки
def make_rows(U, D_end, step=14, mask=None):
    """Строки: УПД и дата t = дата подписания и далее каждые step дней, пока УПД открыт и срок не наступил."""
    close = U.close_d.values
    end = np.minimum(U.plan_d.values, D_end).astype(float)
    end = np.where(np.isnan(close), end, np.minimum(end, close - 1))
    cnt = np.floor((end - U.upd_d.values) / step).astype(np.int64) + 1
    cnt = np.where(end >= U.upd_d.values, cnt, 0)
    if mask is not None:
        cnt = np.where(mask, cnt, 0)
    ui = np.repeat(np.arange(len(U)), cnt)
    j = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    return pd.DataFrame(dict(u=ui, t=U.upd_d.values[ui] + step * j, j=j))


def make_labels(U, rows, D_end):
    """Метки по наблюдаемым данным на D_end: задержка > k (1 / 0 / NaN — не берём) и фактическая задержка закрытых."""
    close = U.close_d.values[rows.u.values]
    plan = U.plan_d.values[rows.u.values]
    closed = ~np.isnan(close)
    dly = np.where(closed, np.maximum(0, close - plan), np.nan)
    over = D_end - plan
    out = pd.DataFrame(index=rows.index)
    for k in KS:
        out["y%d" % k] = np.where(closed, (dly > k).astype(float), np.where(over > k, 1.0, np.nan))
    out["delay_obs"] = dly
    return out


# ------------------------------------------------------------------ история по группам
class Hist:
    """Закрытые УПД группы (пара или покупатель), упорядоченные по дню закрытия; запросы «на дату t» без заглядывания вперёд."""

    def __init__(self, g, cd, upd_d, uid_rank, delay):
        order = np.lexsort((uid_rank, upd_d, cd, g))
        self.g, self.cd, self.dl = g[order], cd[order], delay[order]
        self.keys = self.g.astype(np.int64) * BIG + self.cd.astype(np.int64)
        z = np.zeros(1)
        self.cs = np.concatenate([z, np.cumsum(self.dl)])
        self.cc = {t: np.concatenate([z, np.cumsum(self.dl > t)]) for t in HIST_T}
        self.cmax = pd.Series(self.dl).groupby(self.g).cummax().values if len(self.dl) else np.array([])
        med = np.empty(len(self.dl))
        cur, buf = None, []
        for i, (gg, d) in enumerate(zip(self.g.tolist(), self.dl.tolist())):
            if gg != cur:
                cur, buf = gg, []
            bisect.insort(buf, d)
            m = len(buf)
            med[i] = (buf[(m - 1) // 2] + buf[m // 2]) / 2
        self.med = med

    def query(self, gq, t, prefix):
        gq = gq.astype(np.int64)
        t = t.astype(np.int64)
        base = gq * BIG
        lo = np.searchsorted(self.keys, base, side="left")
        hi = np.searchsorted(self.keys, base + t, side="right")
        n = hi - lo
        ok = n > 0
        nn = np.where(ok, n, 1)
        out = {prefix + "n": n.astype(float)}
        out[prefix + "mean"] = np.where(ok, (self.cs[hi] - self.cs[lo]) / nn, np.nan)
        for th in HIST_T:
            out[prefix + "gt%d" % th] = np.where(ok, (self.cc[th][hi] - self.cc[th][lo]) / nn, np.nan)
        last = np.maximum(hi - 1, 0)
        if len(self.dl):
            out[prefix + "median"] = np.where(ok, self.med[last], np.nan)
            out[prefix + "max"] = np.where(ok, self.cmax[last], np.nan)
            out[prefix + "last"] = np.where(ok, self.dl[last], np.nan)
        else:
            for nme in ("median", "max", "last"):
                out[prefix + nme] = np.full(len(gq), np.nan)
        lo1 = np.searchsorted(self.keys, base + t - 91, side="right")
        lo2 = np.searchsorted(self.keys, base + t - 182, side="right")
        n1, n2 = hi - lo1, lo1 - lo2
        r = np.where(n1 > 0, (self.cs[hi] - self.cs[lo1]) / np.maximum(n1, 1), np.nan)
        q = np.where(n2 > 0, (self.cs[lo1] - self.cs[lo2]) / np.maximum(n2, 1), np.nan)
        out[prefix + "recent3"] = r
        out[prefix + "prev3"] = q
        out[prefix + "trend"] = r - q
        return out


def build_hists(U):
    c = U[~np.isnan(U.close_d.values)]
    if len(c):
        dl = np.maximum(0, c.close_d.values - c.plan_d.values).astype(float)
        rank = pd.Series(c.upd_id.values).rank(method="first").values
        hp = Hist(c.pair.values, c.close_d.values, c.upd_d.values, rank, dl)
        _, bcode = np.unique(c.buyer_inn.values, return_inverse=True)
        hb = Hist(bcode, c.close_d.values, c.upd_d.values, rank, dl)
    else:
        z = np.array([])
        hp = hb = Hist(z.astype(int), z, z, z, z)
        bcode = z.astype(int)
    return hp, hb


def buyer_codes(U):
    """Код покупателя среди всех УПД U (согласован с build_hists только по закрытым — поэтому считаем отдельно)."""
    return pd.factorize(U.buyer_inn.values)[0]


def build_hists_full(U):
    """Истории по паре и по покупателю; коды групп — по всем УПД U, чтобы запросы совпадали."""
    c_mask = ~np.isnan(U.close_d.values)
    c = U[c_mask]
    bc = buyer_codes(U)
    dl = np.maximum(0, c.close_d.values - c.plan_d.values).astype(float)
    rank = pd.Series(c.upd_id.values).rank(method="first").values if len(c) else np.array([])
    hp = Hist(c.pair.values, c.close_d.values, c.upd_d.values, rank, dl)
    hb = Hist(bc[c_mask], c.close_d.values, c.upd_d.values, rank, dl)
    return hp, hb, bc


# ------------------------------------------------------------------ признаки
FEATURES = [
    # УПД
    "amount_mln", "deferral_days", "days_since_signed", "days_to_plan", "paid_share", "plan_month", "plan_qend",
    "plan_yend", "amount_ratio",
    # история пары
    "pair_n", "pair_gt0", "pair_gt14", "pair_gt30", "pair_mean", "pair_median", "pair_max", "pair_last",
    "pair_recent3", "pair_prev3", "pair_trend",
    # история покупателя у всех клиентов партии
    "buy_n", "buy_gt0", "buy_gt14", "buy_gt30", "buy_mean", "buy_median", "buy_max", "buy_last",
    "buy_recent3", "buy_prev3", "buy_trend", "buy_n_clients",
    # покупатель
    "pd", "level", "payables_days", "abs_liquidity", "current_liquidity", "log_revenue", "age_years", "okved",
    "is_moscow",
    # клиент
    "industry"]
CATEGORICAL = ["okved", "industry"]


def amount_ratio(U, idx):
    """Сумма УПД к медиане сумм последних 10 УПД той же пары, подписанных раньше (по порядку дата, номер)."""
    a = U.amount_k.values
    pr = U.pair.values
    out = np.full(len(idx), np.nan)
    first = {}
    for i, g in enumerate(pr):
        if g not in first:
            first[g] = i
    for r, i in enumerate(idx):
        s = first[pr[i]]
        if i > s:
            out[r] = a[i] / np.median(a[max(s, i - 10):i])
    return out


def features(tab, U, A, rows, industries, hists=None):
    """Признаки строк rows (колонки u, t). industries — фиксированный словарь отраслей клиентов."""
    hp, hb, bc = hists if hists is not None else build_hists_full(U)
    ui, t = rows.u.values, rows.t.values
    X = {}
    X["amount_mln"] = U.amount_k.values[ui] / 1e8
    X["deferral_days"] = (U.plan_d.values[ui] - U.upd_d.values[ui]).astype(float)
    X["days_since_signed"] = (t - U.upd_d.values[ui]).astype(float)
    X["days_to_plan"] = (U.plan_d.values[ui] - t).astype(float)
    # оплаченная часть на дату t
    if len(A):
        Ak = A.u.values.astype(np.int64) * BIG + A.d.values
        cs = np.concatenate([[0], np.cumsum(A.a.values)])
        lo = np.searchsorted(Ak, ui.astype(np.int64) * BIG, side="left")
        hi = np.searchsorted(Ak, ui.astype(np.int64) * BIG + t, side="right")
        paid = cs[hi] - cs[lo]
    else:
        paid = np.zeros(len(ui))
    X["paid_share"] = paid / U.amount_k.values[ui]
    pdte = pd.to_datetime(EPOCH + U.plan_d.values[ui].astype("timedelta64[D]"))
    X["plan_month"] = pdte.month.values.astype(float)
    qend = pdte.to_period("Q").end_time.normalize()
    X["plan_qend"] = ((qend - pdte).days.values <= 14).astype(float)
    X["plan_yend"] = ((pdte.month.values == 12) & (pdte.day.values >= 16)).astype(float)
    uniq = np.unique(ui)
    ar = dict(zip(uniq.tolist(), amount_ratio(U, uniq)))
    X["amount_ratio"] = np.array([ar[i] for i in ui.tolist()])
    X.update(hp.query(U.pair.values[ui], t, "pair_"))
    X.update(hb.query(bc[ui], t, "buy_"))
    # число клиентов покупателя в партии, подписавших УПД не позже t
    fu = U.groupby(["buyer_inn", "client_id"], sort=False).upd_d.min().reset_index()
    fu["b"] = fu.buyer_inn.map(dict(zip(U.buyer_inn.values, bc))).values
    fk = np.sort(fu.b.values.astype(np.int64) * BIG + fu.upd_d.values)
    bq = bc[ui].astype(np.int64) * BIG
    X["buy_n_clients"] = (np.searchsorted(fk, bq + t, side="right") - np.searchsorted(fk, bq, side="left")).astype(float)
    # покупатель (снимок на D_end, не меняется во времени)
    by = tab["by"].set_index("buyer_inn")
    inn = U.buyer_inn.values[ui]
    g = by.reindex(inn)
    X["pd"] = g.pd.values
    X["level"] = g.risk_level.map(LEVELS).values.astype(float)
    X["payables_days"] = g.payables_days.values
    X["abs_liquidity"] = g.abs_liquidity.values
    X["current_liquidity"] = g.current_liquidity.values
    X["log_revenue"] = np.log(g.revenue_mln.values.astype(float))
    X["age_years"] = g.age_years.values
    X["okved"] = pd.Categorical(g.okved_section.values, categories=OKVED).codes.astype(float)
    X["is_moscow"] = np.where(g.region.isna().values, np.nan, g.region.isin(MOSCOW).values.astype(float))
    cl = tab["cl"].set_index("client_id")
    X["industry"] = pd.Categorical(cl.industry.reindex(U.client_id.values[ui]).values, categories=industries).codes.astype(float)
    F = pd.DataFrame(X)
    for c in CATEGORICAL:
        F[c] = F[c].where(F[c] >= 0, np.nan)
    return F


def baselines(X):
    """Наивное правило ТЗ: доля УПД с задержкой > k у пары по закрытым УПД; нет истории пары — по покупателю.
    Для expected_days — медиана задержки по паре (нет истории пары — по покупателю)."""
    out = {}
    up, ub = X.pair_n.values > 0, X.buy_n.values > 0
    for k in KS:
        out["naive_p%d" % k] = np.where(up, X["pair_gt%d" % k].values, np.where(ub, X["buy_gt%d" % k].values, np.nan))
    out["naive_days"] = np.where(up, X.pair_median.values, np.where(ub, X.buy_median.values, np.nan))
    return pd.DataFrame(out, index=X.index)
