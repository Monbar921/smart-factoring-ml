"""Final evaluation on cohort 2023 (12m + 21m), acceptance metrics, leakage audit, artifacts."""
import json, pickle, sys, numpy as np, pandas as pd, lightgbm as lgb
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from pd_data import *
from pd_model import *
from pd_metrics import *
from pd_features import NUM_COLS, FEATURES, RU_NAMES

cuts = [float(x) for x in sys.argv[1:4]]
a, b, fb, Xa, Xb = load()
best = json.load(open(OUT + "lgbm_best.json")); iters = int(best["iters"])
S = pickle.load(open(OUT + "05a.pkl", "rb"))
booster = lgb.Booster(model_str=S["booster_txt"])
y_oof = a.y_B_12m.dropna().astype(int).values
iso = fit_calibrator(S["oof_raw"], y_oof)          # Platt on out-of-fold scores (chosen by CV logloss, see calibration_methods.json)
S["oof_pd"] = np.clip(iso.predict(S["oof_raw"]), PD_FLOOR, PD_CAP)
model = PDModel(fb, booster, iso, cuts, meta=dict(calibration="platt", best_params=best, rounds=int(round(iters * 1.1)), train_cohort=2022,
                                                   rfsd_revision=str(a.rfsd_revision.iloc[0]), label="y_B_12m"))
R = {"cuts": cuts}

# ---------- training cohort (OOF) ----------
m = a.y_B_12m.notna().values
yt = a.y_B_12m[m].astype(int).values; oof_pd = S["oof_pd"]
R["train"] = dict(n=int(len(yt)), defaults=int(yt.sum()), rate=float(yt.mean()), oof_gini=gini(yt, S["oof_raw"]))
lv_tr = model.level(oof_pd)
level_table(yt, lv_tr).to_csv(OUT + "levels_train_oof.csv", index=False)
print("TRAIN OOF levels\n", level_table(yt, lv_tr))

# ---------- test cohort 2023 ----------
def evaluate(window, label_col, tag):
    mm = b[label_col].notna().values
    y = b.loc[mm, label_col].astype(int).values
    X = Xb[mm]
    p = model.predict_pd(X); raw = booster.predict(X)
    seg = (b.loc[mm, "line_2110"] >= 800000).values
    out = dict(n=int(len(y)), defaults=int(y.sum()), rate=float(y.mean()), mean_pd=float(p.mean()), gini=gini(y, p))
    out["gini_ci95"] = boot_gini(y, p)
    out["segment_800m"] = dict(n=int(seg.sum()), defaults=int(y[seg].sum()), gini=gini(y[seg], p[seg]), gini_ci95=boot_gini(y[seg], p[seg]))
    dt = decile_table(y, p); dt.to_csv(OUT + f"calibration_deciles_{tag}.csv", index=False)
    lt = level_table(y, model.level(p)); lt.to_csv(OUT + f"levels_{tag}.csv", index=False)
    out["levels"] = lt.to_dict("records")
    out["levels_monotonic"] = bool(lt["default_rate"].is_monotonic_increasing)
    out["decile_max_abs_gap_pp"] = float((dt["mean_pd"] - dt["default_rate"]).abs().max() * 100)
    out["decile_default_rate_monotonic_share"] = float((dt["default_rate"].diff().dropna() >= 0).mean())
    return out, y, p, X, mm

R["test_12m"], y12, p12, X12, m12 = evaluate("12m", "y_B_12m", "2023_12m")
R["test_21m"], y21, p21, X21, m21 = evaluate("21m", "y_B_21m", "2023_21m")
R["psi_score"] = psi(oof_pd, p12, edges=np.quantile(oof_pd, np.linspace(0, 1, 11)))
R["psi_features"] = {c: psi(Xa[c].astype(float), Xb[c].astype(float)) for c in NUM_COLS}
print(json.dumps({k: R[k] for k in ("train", "test_12m", "test_21m", "psi_score")}, indent=1, default=float))

# ---------- leakage audit ----------
audit = {}
# 1) Gini by default type, 2023 and OOF 2022
for nm, d, mask, p, y_all in (("2023_12m", b, m12, p12, y12), ("2022_oof", a, m, oof_pd, yt)):
    reasons = d.loc[mask, "final_reason"].values
    for r in ("банкротство", "исключение"):
        sel = (y_all == 0) | (reasons == r)
        audit[f"gini_{nm}_{r}_vs_nondefault"] = dict(defaults=int((reasons == r).sum()), gini=gini(y_all[sel], p[sel]))
# 2) Gini by default quarter within 12m window (2023 cohort): early vs late defaults
dd = pd.to_datetime(b.loc[m12, "default_date"], errors="coerce").values
q_edges = pd.to_datetime(["2024-04-01", "2024-07-01", "2024-10-01", "2025-01-01", "2025-04-01"])
for i in range(4):
    inq = (dd >= q_edges[i].to_datetime64()) & (dd < q_edges[i + 1].to_datetime64())
    sel = (y12 == 0) | inq
    audit[f"gini_2023_default_quarter_{i+1}"] = dict(defaults=int(inq.sum()), gini=gini(y12[sel], p12[sel]))
# 3) robustness: CV Gini on 2022 without the strongest features (fixed rounds)
Xt = Xa[m].reset_index(drop=True); folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xt, yt))
def cv_without(drop):
    cols = [c for c in FEATURES if c not in drop]; oof = np.zeros(len(yt))
    for tr, va in folds:
        mdl = fit_lgb(Xt.iloc[tr][cols], yt[tr], best, iters); oof[va] = mdl.predict(Xt.iloc[va][cols])
    return gini(yt, oof)
audit["cv_gini_all_features"] = R["train"]["oof_gini"]
audit["cv_gini_without_net_margin"] = cv_without(["net_margin"])
audit["cv_gini_without_net_margin_and_cash_ratio"] = cv_without(["net_margin", "cash_ratio"])
audit["cv_gini_without_age_and_no_prev_report"] = cv_without(["age", "no_prev_report"])
# 4) single-feature Gini on 2023
from sklearn.metrics import roc_auc_score
sf = {}
for c in NUM_COLS:
    x = X12[c].astype(float); ok = x.notna().values
    if ok.sum() > 1000 and len(np.unique(y12[ok])) == 2: sf[c] = 2 * roc_auc_score(y12[ok], x[ok]) - 1
audit["single_feature_gini_2023_top"] = dict(sorted(sf.items(), key=lambda kv: -abs(kv[1]))[:8])
audit["max_single_feature_abs_gini"] = float(max(abs(v) for v in sf.values()))
audit["gini_above_0_9"] = bool(R["test_12m"]["gini"] > 0.9)
R["leakage_audit"] = audit
print(json.dumps(audit, indent=1, default=float))

# ---------- SHAP / importance ----------
sample = np.random.default_rng(1).choice(len(X12), 20000, replace=False)
sv = model.contributions(X12.iloc[sample])[:, :-1]
shap_imp = pd.Series(np.abs(sv).mean(0), index=FEATURES).sort_values(ascending=False)
shap_imp.to_csv(OUT + "shap_mean_abs.csv", header=["mean_abs_shap"])
gain = pd.Series(booster.feature_importance("gain"), index=booster.feature_name()).sort_values(ascending=False)
(gain / gain.sum()).to_csv(OUT + "gain_importance.csv", header=["share_gain"])
R["top_shap"] = shap_imp.head(10).round(4).to_dict()

# ---------- plots ----------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for tag, c in (("2023_12m", "#1f77b4"), ("2023_21m", "#d62728")):
    dt = pd.read_csv(OUT + f"calibration_deciles_{tag}.csv")
    ax[0].plot(dt.mean_pd * 100, dt.default_rate * 100, "o-", color=c, label=tag.replace("2023_", "2023, окно "))
mx = max(pd.read_csv(OUT + "calibration_deciles_2023_12m.csv").default_rate.max(), 0.05) * 100
ax[0].plot([0, mx], [0, mx], "k--", lw=1, label="идеальная калибровка"); ax[0].set_xlabel("Средний PD в дециле, %"); ax[0].set_ylabel("Фактическая доля дефолтов, %")
ax[0].set_title("Калибровка по децилям"); ax[0].legend(); ax[0].grid(alpha=.3)
lt = pd.read_csv(OUT + "levels_2023_12m.csv"); lt1 = pd.read_csv(OUT + "levels_train_oof.csv")
w = .38; xs = np.arange(4)
ax[1].bar(xs - w/2, lt1.default_rate * 100, w, label="2022 (вне обучения)", color="#9ecae1"); ax[1].bar(xs + w/2, lt.default_rate * 100, w, label="2023 (проверка)", color="#1f77b4")
ax[1].set_xticks(xs); ax[1].set_xticklabels(lt.level); ax[1].set_ylabel("Доля дефолтов, %"); ax[1].set_title("Уровни риска A–D"); ax[1].legend(); ax[1].grid(alpha=.3, axis="y")
top = shap_imp.head(12)[::-1]
ax[2].barh([RU_NAMES.get(i, i) for i in top.index], top.values, color="#2ca02c"); ax[2].set_title("Топ признаков по SHAP (среднее |вклад|)"); ax[2].grid(alpha=.3, axis="x")
plt.tight_layout(); plt.savefig(OUT + "pd_report_plots.png", dpi=130); plt.close()

# ---------- artifacts ----------
model.save(OUT + "pd_model.pkl")
json.dump(R, open(OUT + "pd_results.json", "w"), indent=1, default=float, ensure_ascii=False)
print("DONE")
