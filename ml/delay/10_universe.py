"""Universe of real buyers for the delay generator: store FY2024 + PD/level + status as of D_end derived from labels."""
import os, sys, json
import numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-claude/bcdabce6-a474-53a6-84e4-d4a6f0c4f498/scratchpad"
sys.path.insert(0, f"{SP}/pdpkg")
os.environ["PD_MODEL_PATH"] = f"{SP}/pdpkg/out/pd_model.pkl"
from pd_model import PDModel
D_END = pd.Timestamp("2025-12-30")

m = PDModel.load(os.environ["PD_MODEL_PATH"])
s = pd.read_parquet(f"{SP}/pdpkg/data/scoring_store_2024.parquet")
X = m.features(s)
s["pd"] = m.predict_pd(X)
s["level"] = m.level(s["pd"].values)
print(s.level.value_counts(normalize=True).round(3).to_dict())

lab = pd.read_csv(f"{SP}/ans/final/send/cohort_2023_labels.csv", dtype=str, low_memory=False)
for c in ("default_date", "egrul_event_date", "efrsb_episode_start"):
    lab[c] = pd.to_datetime(lab[c], errors="coerce")
lab["in_labels"] = 1
keep = ["inn","final_reason","default_date","egrul_reason","egrul_event_date","efrsb_episode_start","efrsb_case_count","efrsb_checked","manual_check","in_labels"]
u = s.merge(lab[keep], on="inn", how="left")
u["in_labels"] = u["in_labels"].fillna(0).astype(int)

def status_asof(r):
    """returns stop type / process flags as of D_END from raw label fields"""
    bank = pd.notna(r.efrsb_episode_start) and r.efrsb_episode_start <= D_END
    er = r.egrul_reason if isinstance(r.egrul_reason, str) else ""
    ed = r.egrul_event_date
    eg = pd.notna(ed) and ed <= D_END
    return bank, er, eg
# vectorised
bank = u.efrsb_episode_start.le(D_END)
er = u.egrul_reason.fillna("")
eg = u.egrul_event_date.le(D_END)
u["asof_bankruptcy"] = bank.fillna(False)
u["asof_egrul_event"] = eg.fillna(False)
u["asof_egrul_reason"] = np.where(u.asof_egrul_event, er, "")
fr = u.final_reason.fillna("")
dd = u.default_date
u["pre_cutoff"] = fr.eq("до среза")
# any final-reason event dated <= D_END
u["asof_final_event"] = dd.le(D_END).fillna(False)
u["stop_type"] = ""
u.loc[u.asof_bankruptcy | (fr.eq("банкротство") & u.asof_final_event), "stop_type"] = "банкротство"
m_liq = (fr.isin(["добровольная", "принудительная ликвидация"]) & u.asof_final_event) | u.asof_egrul_reason.str.contains("ликвидац")
m_exc = (fr.eq("исключение") & u.asof_final_event) | u.asof_egrul_reason.str.startswith("исключение")
u.loc[(u.stop_type == "") & m_liq, "stop_type"] = "ликвидация"
u.loc[(u.stop_type == "") & m_exc, "stop_type"] = "исключение"
m_reorg = (fr.eq("реорганизация") & u.asof_final_event) | u.asof_egrul_reason.str.startswith("реорганизация")
u["reorg"] = m_reorg
u["clean_asof"] = (u.in_labels == 1) & ~u.pre_cutoff & (u.stop_type == "") & ~u.reorg & ~u.asof_egrul_event & ~u.asof_bankruptcy & u.manual_check.isna()
print("in store:", len(u), " with labels:", u.in_labels.sum(), " clean as of D_END:", int(u.clean_asof.sum()))
print(u.stop_type.value_counts().to_dict(), " reorg:", int(u.reorg.sum()), " pre_cutoff:", int(u.pre_cutoff.sum()))
u.to_parquet(f"{SP}/buyers/universe_store_2024.parquet", index=False)
