import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

def gini(y, p):
    return 2 * roc_auc_score(y, p) - 1

def boot_gini(y, p, n=1000, seed=7):
    """Bootstrap 95% CI of Gini."""
    y = np.asarray(y); p = np.asarray(p)
    rng = np.random.default_rng(seed); out = []
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])   # stratified resampling
        out.append(2 * roc_auc_score(y[idx], p[idx]) - 1)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

def decile_table(y, p, q=10):
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["decile"] = pd.qcut(d["p"].rank(method="first"), q, labels=False) + 1
    t = d.groupby("decile").agg(n=("y", "size"), mean_pd=("p", "mean"), default_rate=("y", "mean"), defaults=("y", "sum")).reset_index()
    return t

def psi(expected, actual, bins=10, edges=None):
    expected = pd.Series(expected).dropna(); actual = pd.Series(actual).dropna()
    if edges is None:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges = np.asarray(edges, dtype=float); edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected); a = np.histogram(actual, edges)[0] / len(actual)
    e = np.clip(e, 1e-6, None); a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))

def level_table(y, level, order="ABCD"):
    d = pd.DataFrame({"y": np.asarray(y), "level": np.asarray(level)})
    t = d.groupby("level").agg(n=("y", "size"), defaults=("y", "sum"), default_rate=("y", "mean")).reindex(list(order)).reset_index()
    t["share"] = t["n"] / t["n"].sum()
    return t
