"""Compare label variants B / A / 0 by 5-fold CV inside cohort 2022 (same model, same features, same folds)."""
import json, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from pd_data import *
from pd_model import *
from pd_metrics import *
a, b, fb, Xa, Xb = load()
best = json.load(open(OUT + "lgbm_best.json")); iters = int(best["iters"])
V = {"B": "y_B_12m", "A": "y_A_12m", "0": "y_0_12m"}
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xa, a["y_0_12m"].astype(int)))
oof = {k: np.zeros(len(a)) for k in V}
for k, col in V.items():
    ok = a[col].notna().values
    for tr, va in folds:
        trn = tr[ok[tr]]
        mdl = fit_lgb(Xa.iloc[trn], a[col].iloc[trn].astype(int), best, iters)
        oof[k][va] = mdl.predict(Xa.iloc[va])
    print("trained variant", k, flush=True)
# evaluation targets, each on its own defined rows:
#   T_B = "any default" (bankruptcy + exclusion), rows with y_B defined
#   T_A = "bankruptcy only", rows with y_A defined (a subset of the y_B rows: exclusion cases are removed there)
T = {"target B (bankruptcy + exclusion)": "y_B_12m", "target A (bankruptcy only)": "y_A_12m"}
rows = []
for tv in V:
    for tn, tc in T.items():
        ok = a[tc].notna().values; y = a.loc[ok, tc].astype(int).values
        p = oof[tv][ok]; lo, hi = boot_gini(y, p, n=300)
        fg = [gini(a.loc[ok, tc].astype(int).values[np.isin(np.where(ok)[0], va)], p[np.isin(np.where(ok)[0], va)]) for _, va in folds]
        rows.append(dict(trained_on=tv, evaluated_against=tn, n=int(ok.sum()), defaults=int(y.sum()), gini=gini(y, p),
                         ci_low=lo, ci_high=hi, fold_std=float(np.std(fg))))
res = pd.DataFrame(rows); res.to_csv(OUT + "label_variants_cv.csv", index=False)
print(res.round(4).to_string())
