import pandas as pd, numpy as np, json, time
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer, SplineTransformer
from sklearn.metrics import roc_auc_score
from pd_data import *
from pd_features import NUM_COLS, FLAG_COLS, CAT_COLS

def make_lr(miss_cols, spline, C):
    num_steps = [("imp", SimpleImputer(strategy="median", add_indicator=False))]
    if spline:
        num_steps += [("q", QuantileTransformer(n_quantiles=200, output_distribution="normal", subsample=200000, random_state=SEED)),
                      ("sp", SplineTransformer(n_knots=5, degree=3, include_bias=False))]
    else:
        num_steps += [("sc", StandardScaler())]
    ct = ColumnTransformer([
        ("num", Pipeline(num_steps), NUM_COLS),
        ("miss", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=0.0, add_indicator=True)),
                           ("drop", "passthrough")]), []),
        ("flag", "passthrough", FLAG_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=100), CAT_COLS),
    ])
    return ct, LogisticRegression(C=C, class_weight="balanced", max_iter=1000, solver="lbfgs")

class LR:
    """Logistic regression with missing indicators (for columns with >2% NaN in training data)."""
    def __init__(self, spline, C): self.spline, self.C = spline, C
    def _prep(self, X):
        X = X.copy()
        for c in self.miss_cols: X["miss_" + c] = X[c].isna().astype(int)
        for c in CAT_COLS: X[c] = X[c].astype(str)
        return X
    def fit(self, X, y):
        self.miss_cols = [c for c in NUM_COLS if X[c].isna().mean() > 0.02]
        ct, lr = make_lr(self.miss_cols, self.spline, self.C)
        ct.transformers[2] = ("flag", "passthrough", FLAG_COLS + ["miss_" + c for c in self.miss_cols])
        ct.transformers[1] = ("miss", "drop", [])
        self.ct, self.lr = ct, lr
        Z = ct.fit_transform(self._prep(X)); lr.fit(Z, y); return self
    def predict(self, X):
        return self.lr.predict_proba(self.ct.transform(self._prep(X)))[:, 1]

if __name__ == "__main__":
    a, b, fb, Xa, Xb = load()
    m = a.y_B_12m.notna().values
    X, y = Xa[m].reset_index(drop=True), a.y_B_12m[m].astype(int).reset_index(drop=True)
    from sklearn.model_selection import StratifiedKFold
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
    rows = []
    for spline in (False, True):
        for C in (0.003, 0.03, 0.3):
            t = time.time(); aucs = []
            for tr, va in folds:
                mdl = LR(spline, C).fit(X.iloc[tr], y.iloc[tr])
                aucs.append(roc_auc_score(y.iloc[va], mdl.predict(X.iloc[va])))
            r = dict(spline=spline, C=C, mean_gini=2 * np.mean(aucs) - 1, std_gini=2 * np.std(aucs), sec=round(time.time() - t))
            rows.append(r); print(r, flush=True)
    res = pd.DataFrame(rows).sort_values("mean_gini", ascending=False)
    res.to_csv(OUT + "logreg_search.csv", index=False)
    json.dump(res.iloc[0].to_dict(), open(OUT + "logreg_best.json", "w"), indent=1, default=float)
    print("BEST", res.iloc[0].to_dict())
