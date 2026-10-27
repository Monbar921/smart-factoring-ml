"""OOF predictions on cohort 2022 (5-fold, fixed rounds), calibration, final booster on all of 2022."""
import json, pickle, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss
from pd_data import *
from pd_model import *
a, b, fb, Xa, Xb = load()
best = json.load(open(OUT + "lgbm_best.json")); iters = int(best["iters"])
m = a.y_B_12m.notna().values
Xt, yt = Xa[m].reset_index(drop=True), a.y_B_12m[m].astype(int).reset_index(drop=True)
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xt, yt))
oof_raw = np.zeros(len(Xt)); fold_id = np.zeros(len(Xt), int)
for k, (tr, va) in enumerate(folds):
    mdl = fit_lgb(Xt.iloc[tr], yt.iloc[tr], best, iters)
    oof_raw[va] = mdl.predict(Xt.iloc[va]); fold_id[va] = k
    print("fold", k, "gini", round(gini(yt.iloc[va], oof_raw[va]), 4), flush=True)
print("OOF gini", gini(yt, oof_raw))

# calibration choice: cross-fitted isotonic vs Platt (fit on 4 folds of OOF, scored on the 5th)
cf = {"isotonic": np.zeros(len(yt)), "platt": np.zeros(len(yt))}
logit = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
for k in range(5):
    tr, va = fold_id != k, fold_id == k
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(oof_raw[tr], yt[tr])
    cf["isotonic"][va] = np.clip(iso.predict(oof_raw[va]), PD_FLOOR, PD_CAP)
    pl = LogisticRegression(C=1e6, max_iter=500).fit(logit(oof_raw[tr]).reshape(-1, 1), yt[tr])
    cf["platt"][va] = pl.predict_proba(logit(oof_raw[va]).reshape(-1, 1))[:, 1]
calib_cmp = {k: dict(logloss=float(log_loss(yt, v)), brier=float(brier_score_loss(yt, v)), mean_pd=float(v.mean())) for k, v in cf.items()}
calib_cmp["raw_uncalibrated"] = dict(logloss=float(log_loss(yt, np.clip(oof_raw, 1e-6, 1 - 1e-6))), brier=float(brier_score_loss(yt, oof_raw)), mean_pd=float(oof_raw.mean()))
print(json.dumps(calib_cmp, indent=1))
iso = fit_calibrator(oof_raw, yt)
oof_pd = np.clip(iso.predict(oof_raw), PD_FLOOR, PD_CAP)
print("quantiles of OOF calibrated PD:", {q: round(float(np.quantile(oof_pd, q)), 5) for q in (0.1, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)})
booster = fit_lgb(Xt, yt, best, int(round(iters * 1.1)))
pickle.dump(dict(oof_raw=oof_raw, oof_pd=oof_pd, fold_id=fold_id, iso=iso, booster_txt=booster.model_to_string(), calib_cmp=calib_cmp,
                 idx=np.where(m)[0]), open(OUT + "05a.pkl", "wb"))
json.dump(calib_cmp, open(OUT + "calibration_methods.json", "w"), indent=1)
