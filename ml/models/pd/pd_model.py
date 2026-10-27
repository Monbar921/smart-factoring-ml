"""
PD model wrapper: feature builder + LightGBM + isotonic calibration + risk levels A-D + SHAP top-3.
"""
import numpy as np, pandas as pd, lightgbm as lgb, pickle
from sklearn.linear_model import LogisticRegression
from pd_features import FEATURES, RU_NAMES, FeatureBuilder

PD_FLOOR, PD_CAP = 0.0005, 0.95        # calibrated PD is clipped to this range
LEVELS = ["A", "B", "C", "D"]


def lgb_params(best, seed=42, threads=2):
    return dict(objective="binary", learning_rate=0.05, num_leaves=int(best["num_leaves"]),
                min_child_samples=int(best["min_child_samples"]), feature_fraction=float(best["feature_fraction"]),
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=float(best["lambda_l2"]), verbose=-1,
                num_threads=threads, seed=seed)


def fit_lgb(X, y, best, rounds, seed=42):
    w = np.where(np.asarray(y) == 1, float(best["pos_weight"]), 1.0)
    return lgb.train(lgb_params(best, seed), lgb.Dataset(X, np.asarray(y), weight=w), int(rounds))


class PlattCalibrator:
    """Platt scaling: PD = sigmoid(a * logit(raw_score) + b), fitted on out-of-fold scores."""
    def fit(self, raw, y):
        z = self._logit(raw).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, np.asarray(y))
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
        return np.log(p / (1 - p))

    def predict(self, raw):
        return 1 / (1 + np.exp(-(self.a * self._logit(raw) + self.b)))


def fit_calibrator(raw_oof, y):
    return PlattCalibrator().fit(raw_oof, y)


def assign_level(pd_, cuts):
    """cuts = [c1, c2, c3]: A < c1 <= B < c2 <= C < c3 <= D"""
    return np.select([pd_ < cuts[0], pd_ < cuts[1], pd_ < cuts[2]], LEVELS[:3], default="D")


class PDModel:
    def __init__(self, fb: FeatureBuilder, booster: lgb.Booster, calibrator, cuts, meta=None):
        self.fb, self.booster, self.calibrator, self.cuts, self.meta = fb, booster, calibrator, list(cuts), meta or {}

    # ---- scoring -------------------------------------------------------------------------------
    def features(self, raw: pd.DataFrame) -> pd.DataFrame:
        return self.fb.transform(raw)

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.booster.predict(X)
        return np.clip(self.calibrator.predict(raw), PD_FLOOR, PD_CAP)

    def level(self, pd_):
        return assign_level(np.asarray(pd_), self.cuts)

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """SHAP values (log-odds) per feature; last column is the base value."""
        return self.booster.predict(X, pred_contrib=True)

    def top_factors(self, X: pd.DataFrame, k=3):
        sv = self.contributions(X)[:, :-1]
        names = list(X.columns)
        out = []
        for i in range(len(X)):
            idx = np.argsort(-np.abs(sv[i]))[:k]
            fl = []
            for j in idx:
                v = X.iloc[i, j]
                fl.append(dict(feature=names[j], name_ru=RU_NAMES.get(names[j], names[j]),
                               value=(None if (isinstance(v, float) and np.isnan(v)) else (v if isinstance(v, (str, int, float, np.integer, np.floating)) else str(v))),
                               direction="raises_risk" if sv[i, j] > 0 else "lowers_risk",
                               shap=float(sv[i, j])))
            out.append(fl)
        return out

    def score_frame(self, raw: pd.DataFrame, k=3) -> pd.DataFrame:
        X = self.features(raw)
        p = self.predict_pd(X)
        res = pd.DataFrame({"inn": raw["inn"].values, "pd": p, "level": self.level(p)})
        facs = self.top_factors(X, k)
        for i in range(k):
            res[f"factor_{i+1}"] = [f[i]["name_ru"] + (" ↑риск" if f[i]["direction"] == "raises_risk" else " ↓риск") for f in facs]
        return res

    # ---- persistence ---------------------------------------------------------------------------
    def save(self, path):
        state = dict(fb=self.fb, booster_txt=self.booster.model_to_string(), calibrator=self.calibrator,
                     cuts=self.cuts, meta=self.meta)
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            s = pickle.load(f)
        return PDModel(s["fb"], lgb.Booster(model_str=s["booster_txt"]), s["calibrator"], s["cuts"], s["meta"])
