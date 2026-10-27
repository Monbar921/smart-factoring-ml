"""Independent re-check of the headline numbers: fresh process, saved model, rank-based Gini, counts vs TZ tables."""
import json, numpy as np, pandas as pd
from scipy.stats import rankdata
from pd_data import *
from pd_model import PDModel
from pd_metrics import psi
R = json.load(open(OUT + "pd_results.json"))
M = PDModel.load(OUT + "pd_model.pkl")
def gini_rank(y, p):
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return 2 * ((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) - 1
chk = {}
b = pd.read_parquet(U + "cohort_2023_raw_features.parquet").merge(pd.read_csv(L + "cohort_2023_labels.csv", dtype={"inn": str}, low_memory=False), on="inn", validate="1:1")
a = pd.read_parquet(U + "cohort_2022_raw_features.parquet").merge(pd.read_csv(L + "cohort_2022_labels.csv", dtype={"inn": str}, low_memory=False), on="inn", validate="1:1")
# 1. counts vs TZ section 4
tz = {("2022", "y_B_12m"): (2182, 172169, 8233), ("2023", "y_B_12m"): (2523, 201151, 6837), ("2023", "y_B_21m"): (5575, 198099, 6837),
      ("2022", "y_A_12m"): (828, 168544, 13212), ("2023", "y_A_12m"): (820, 199076, 10615), ("2022", "y_0_12m"): (2454, 180130, 0), ("2023", "y_0_12m"): (3175, 207336, 0)}
for (c, col), exp in tz.items():
    d = a if c == "2022" else b
    got = (int((d[col] == 1).sum()), int((d[col] == 0).sum()), int(d[col].isna().sum()))
    chk[f"counts_{c}_{col}"] = "OK" if got == exp else f"MISMATCH got {got} expected {exp}"
# 2. headline Gini by rank formula through the saved model (fresh path)
mb = b.y_B_12m.notna().values; y = b.y_B_12m[mb].astype(int).values
X = M.features(b[mb]); p = M.predict_pd(X)
g = gini_rank(y, p); chk["gini_2023_12m_rank_formula"] = round(g, 4); chk["gini_2023_12m_reported"] = round(R["test_12m"]["gini"], 4)
chk["gini_match"] = bool(abs(g - R["test_12m"]["gini"]) < 1e-3)
m21 = b.y_B_21m.notna().values; y21 = b.y_B_21m[m21].astype(int).values
g21 = gini_rank(y21, M.predict_pd(M.features(b[m21]))); chk["gini_21m_match"] = bool(abs(g21 - R["test_21m"]["gini"]) < 1e-3)
seg = (b.loc[mb, "line_2110"] >= 800000).values; gs = gini_rank(y[seg], p[seg]); chk["gini_segment_match"] = bool(abs(gs - R["test_12m"]["segment_800m"]["gini"]) < 1e-3)
# 3. level table by plain pandas
lv = np.where(p < .004, "A", np.where(p < .015, "B", np.where(p < .06, "C", "D")))
t = pd.DataFrame({"y": y, "l": lv}).groupby("l").y.agg(["size", "sum", "mean"])
chk["levels_2023_12m"] = {k: [int(r["size"]), int(r["sum"]), round(float(r["mean"]), 5)] for k, r in t.iterrows()}
chk["levels_monotonic"] = bool(t["mean"].is_monotonic_increasing)
# 4. no post-cutoff columns in the extracts and no forbidden features
leak = ["dissolution_date", "cutoff", "label_end", "legal_default_12m", "legal_default_status", "legal_default_event_type", "legal_default_event_date",
        "label_source", "registered_cessation_12m", "registered_cessation_status", "registered_cessation_event_date", "observation_complete", "censor_reason"]
chk["leak_columns_in_extract_2022"] = [c for c in leak if c in pd.read_parquet(U + "cohort_2022_raw_features.parquet").columns]
chk["model_features"] = list(M.booster.feature_name())
# 5. production drift: FY2024 scoring store vs 2022 training (score and levels)
s = pd.read_parquet(U + "scoring_store_2024.parquet")
p24 = M.predict_pd(M.features(s)); m22 = a.y_B_12m.notna().values; p22 = M.predict_pd(M.features(a[m22]))
chk["psi_score_2024_vs_2022"] = round(psi(p22, p24, edges=np.quantile(p22, np.linspace(0, 1, 11))), 4)
chk["levels_share_2024"] = {k: round(float((M.level(p24) == k).mean()), 4) for k in "ABCD"}
chk["mean_pd_2024"] = round(float(p24.mean()), 5)
json.dump(chk, open(OUT + "verify_pd.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(chk, indent=1, ensure_ascii=False))
