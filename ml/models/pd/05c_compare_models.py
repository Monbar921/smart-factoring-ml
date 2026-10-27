"""Model comparison (LightGBM vs CatBoost vs logistic regression) + corrected Gini-by-default-type audit."""
import json, pickle, numpy as np, pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from pd_data import *
from pd_model import *
from pd_metrics import *
from pd_features import CAT_COLS
import importlib; lrmod = importlib.import_module("03_logreg_cv")
a, b, fb, Xa, Xb = load()
S = pickle.load(open(OUT + "05a.pkl", "rb")); R = json.load(open(OUT + "pd_results.json"))
m = a.y_B_12m.notna().values
Xt, yt = Xa[m].reset_index(drop=True), a.y_B_12m[m].astype(int).values
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xt, yt))
mb = b.y_B_12m.notna().values; m21 = b.y_B_21m.notna().values
y12, y21 = b.y_B_12m[mb].astype(int).values, b.y_B_21m[m21].astype(int).values
rows = []
def add(name, fold_gini, p12, p21):
    r = dict(model=name, cv_gini_mean=float(np.mean(fold_gini)), cv_gini_std=float(np.std(fold_gini)),
             test12_gini=gini(y12, p12), test21_gini=gini(y21, p21))
    r["test12_ci_low"], r["test12_ci_high"] = boot_gini(y12, p12, n=300)
    rows.append(r); print(r, flush=True); return r
# LightGBM (already trained)
fg_lgb = [gini(yt[va], S["oof_raw"][va]) for _, va in folds]
bst = lgb.Booster(model_str=S["booster_txt"]) if False else None
import lightgbm as lgb
bst = lgb.Booster(model_str=S["booster_txt"])
add("LightGBM", fg_lgb, bst.predict(Xb[mb]), bst.predict(Xb[m21]))
# Logistic regression (linear and spline)
fold_gini = {}
for name, spline in (("LogReg linear", False), ("LogReg splines", True)):
    fg = []
    for tr, va in folds:
        mdl = lrmod.LR(spline, 0.03).fit(Xt.iloc[tr], yt[tr]); fg.append(gini(yt[va], mdl.predict(Xt.iloc[va])))
    fold_gini[name] = fg
    mdl = lrmod.LR(spline, 0.03).fit(Xt, yt)
    add(name, fg, mdl.predict(Xb[mb]), mdl.predict(Xb[m21]))
# CatBoost
cb = json.load(open(OUT + "catboost_best.json"))
Xc = Xt.copy(); Xbc = Xb.copy()
for c in CAT_COLS:
    Xc[c] = Xc[c].astype(object).fillna("NA").astype(str); Xbc[c] = Xbc[c].astype(object).fillna("NA").astype(str)
fg = []
cbm = CatBoostClassifier(iterations=int(cb["iters"] * 1.1), learning_rate=0.08, depth=int(cb["depth"]), l2_leaf_reg=float(cb["l2"]),
                         auto_class_weights="SqrtBalanced", random_seed=SEED, thread_count=2, verbose=False, cat_features=CAT_COLS).fit(Xc, yt)
add("CatBoost", [cb["mean_gini"]] * 5, cbm.predict_proba(Xbc[mb])[:, 1], cbm.predict_proba(Xbc[m21])[:, 1])
rows[-1]["note"] = "CV from search (early stopping per fold); fold values not stored"
res = pd.DataFrame(rows)
# paired fold differences
for name in fold_gini:
    d = np.array(fg_lgb) - np.array(fold_gini[name]); res.loc[res.model == name, "lgbm_minus_this_cv_mean"] = d.mean()
    res.loc[res.model == name, "lgbm_better_in_folds"] = f"{int((d > 0).sum())}/5"
res.to_csv(OUT + "model_comparison.csv", index=False); print(res.round(4).to_string())

# ---- corrected audit: Gini by default type ----
aud = {}
for nm, d, mask, p, y in (("2023_12m", b, mb, bst.predict(Xb[mb]), y12), ("2022_oof", a, m, S["oof_raw"], yt)):
    pp = np.clip(fit_calibrator(S["oof_raw"], yt).predict(p), PD_FLOOR, PD_CAP)
    reasons = d.loc[mask, "final_reason"].values
    for r in ("банкротство", "исключение"):
        pos = (y == 1) & (reasons == r); sel = (y == 0) | pos
        aud[f"{nm}__{r}"] = dict(defaults=int(pos.sum()), gini=gini(y[sel], pp[sel]))
R["leakage_audit"] = {k: v for k, v in R["leakage_audit"].items() if not k.startswith("gini_2023_12m_") and not k.startswith("gini_2022_oof_")}
R["leakage_audit"]["gini_by_default_type"] = aud
R["model_comparison"] = res.to_dict("records")
json.dump(R, open(OUT + "pd_results.json", "w"), indent=1, default=float, ensure_ascii=False)
print(json.dumps(aud, indent=1, ensure_ascii=False))
