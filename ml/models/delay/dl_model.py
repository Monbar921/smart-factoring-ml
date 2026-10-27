"""Модель задержки оплаты по УПД: бустинги по p_7, p_14, p_30, p_90, калибровка, упорядочивание, ожидаемые дни."""
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

KS = (7, 14, 30, 90)
VERSION = "delay-1.0"


class Calibrator:
    """Калибровка вероятности: Platt (логистическая по logit) или isotonic (ступенчатая, np.interp). Без sklearn при применении."""

    def __init__(self, kind, a=None, b=None, x=None, y=None):
        self.kind, self.a, self.b, self.x, self.y = kind, a, b, x, y

    @staticmethod
    def fit(kind, p, y):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        if kind == "none":
            return Calibrator("none")
        if kind == "platt":
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(C=1e6, max_iter=500).fit(np.log(p / (1 - p)).reshape(-1, 1), y)
            return Calibrator("platt", a=float(lr.coef_[0, 0]), b=float(lr.intercept_[0]))
        from sklearn.isotonic import IsotonicRegression
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-5, y_max=1 - 1e-5).fit(p, y)
        return Calibrator("isotonic", x=ir.X_thresholds_.copy(), y=ir.y_thresholds_.copy())

    def apply(self, p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        if self.kind == "none":
            return p
        if self.kind == "platt":
            return 1 / (1 + np.exp(-(self.a * np.log(p / (1 - p)) + self.b)))
        return np.interp(p, self.x, self.y)


def order_probs(P):
    """p_7 >= p_14 >= p_30 >= p_90: упорядочивание значений по строке."""
    return np.sort(P, axis=1)[:, ::-1]


class DelayModel:
    def __init__(self, features, categorical, industries, boosters, calibrators, reg, meta):
        self.features, self.categorical, self.industries = features, categorical, industries
        self.boosters, self.calibrators, self.reg, self.meta = boosters, calibrators, reg, meta
        self.version = VERSION
        self._cache = {}

    def _b(self, key, s):
        if key not in self._cache:
            self._cache[key] = lgb.Booster(model_str=s)
        return self._cache[key]

    def raw(self, X):
        Z = X[self.features]
        return {k: self._b(("c", k), self.boosters[k]).predict(Z) for k in KS}

    def predict(self, X):
        R = self.raw(X)
        P = np.column_stack([self.calibrators[k].apply(R[k]) for k in KS])
        P = order_probs(P)
        Z = X[self.features]
        e = self._b("reg", self.reg["model"]).predict(Z) + self.reg.get("shift", 0.0)
        out = pd.DataFrame(P, columns=["p_%d" % k for k in KS], index=X.index)
        out["expected_days"] = np.maximum(e, 0.0)
        return out

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(dict(features=self.features, categorical=self.categorical, industries=self.industries,
                             boosters=self.boosters, calibrators=self.calibrators, reg=self.reg, meta=self.meta,
                             version=self.version), f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        m = DelayModel(d["features"], d["categorical"], d["industries"], d["boosters"], d["calibrators"], d["reg"], d["meta"])
        m.version = d["version"]
        return m
