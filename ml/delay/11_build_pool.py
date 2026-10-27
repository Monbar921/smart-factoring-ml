"""
Buyer pool for the delay generator (TZ delay v1.1, section 4).
1 500 real buyers = 5 blocks x 300. One block = the buyer set of one 100-client batch:
    276 ordinary (A 82 / B 76 / C 66 / D 52), 12 'мало данных' (5 no report + 5 revenue < 120 mln + 2 age < 1),
    9 stop-factor (3 bankruptcy + 3 liquidation + 3 exclusion), 3 'нет данных'.
Statuses as of D_END come from cohort-2023 labels (ЕГРЮЛ + ЕФРСБ).
"""
import os, sys, json
import numpy as np, pandas as pd

SP = "/tmp/claude-0/-home-claude/bcdabce6-a474-53a6-84e4-d4a6f0c4f498/scratchpad"
UP = "/mnt/user-data/uploads/Downloads/pd_start/pd_work"
OUT = f"{SP}/buyers/out"; os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, f"{SP}/pdpkg")
from pd_features import raw_ratios

D_END = pd.Timestamp("2025-12-30")
WIN_START = pd.Timestamp("2023-04-01")
SEED = 42
rng = np.random.default_rng(SEED)
N_BLOCKS = 5
ORD_PER_LEVEL = {"A": 82, "B": 76, "C": 66, "D": 52}          # per block
MD_PER_BLOCK = {"no_report": 5, "revenue_below_120m": 5, "age_below_1": 2}
STOP_PER_TYPE = 3                                            # per block
NO_DATA_PER_BLOCK = 3

SECT = {s: s for s in "GCFMHJLA"}
sector_group = lambda s: s.map(lambda x: SECT.get(x, "other"))
region_group = lambda r: np.select([r == "moscow city", r == "moscow reg.", r == "sankt-petersburg"],
                                   ["moscow_city", "moscow_region", "spb"], "other")
size_band = lambda rev_mln: pd.cut(rev_mln, [0, 300, 1000, 5000, 1e12], labels=["120-300", "300-1000", "1000-5000", "5000+"], right=False).astype(str)
age_band = lambda a: pd.cut(a, [-1, 2, 9, 19, 200], labels=["1-2", "3-9", "10-19", "20+"]).astype(str)


def ipf(df, targets, iters=30):
    """iterative proportional fitting: weights so that weighted shares match targets for every dimension"""
    w = np.ones(len(df))
    for _ in range(iters):
        for col, tgt in targets.items():
            share = pd.Series(w).groupby(df[col].values).sum()
            share = share / share.sum()
            f = {k: (tgt[k] / share[k] if share.get(k, 0) > 0 else 1.0) for k in tgt}
            w = w * df[col].map(f).fillna(1.0).values
    return w / w.sum()


def draw(df, n, p):
    idx = rng.choice(len(df), size=n, replace=False, p=p)
    return df.iloc[idx]


def flat(series, power=0.5):
    s = series.value_counts(normalize=True) ** power
    return (s / s.sum()).to_dict()


# ---------------- labels: status as of D_END ----------------
lab = pd.read_csv(f"{SP}/ans/final/send/cohort_2023_labels.csv", dtype=str, low_memory=False)
for c in ("default_date", "egrul_event_date", "efrsb_episode_start"):
    lab[c] = pd.to_datetime(lab[c], errors="coerce")
fr = lab.final_reason.fillna("")
er = lab.egrul_reason.fillna("")
lab["bank_asof"] = lab.efrsb_episode_start.le(D_END).fillna(False)
lab["egrul_asof"] = lab.egrul_event_date.le(D_END).fillna(False)
fin_asof = lab.default_date.le(D_END).fillna(False)
lab["stop_type"] = ""
lab.loc[lab.bank_asof | (fr.eq("банкротство") & fin_asof), "stop_type"] = "банкротство"
liq = (fr.isin(["добровольная", "принудительная ликвидация"]) & fin_asof) | (lab.egrul_asof & er.str.contains("ликвидац"))
exc = (fr.eq("исключение") & fin_asof) | (lab.egrul_asof & er.str.startswith("исключение"))
lab.loc[(lab.stop_type == "") & liq, "stop_type"] = "ликвидация"
lab.loc[(lab.stop_type == "") & exc, "stop_type"] = "исключение"
reorg = (fr.eq("реорганизация") & fin_asof) | (lab.egrul_asof & er.str.startswith("реорганизация"))
lab["clean"] = (~fr.eq("до среза")) & lab.stop_type.eq("") & ~reorg & ~lab.egrul_asof & ~lab.bank_asof & lab.manual_check.isna()
lab["efrsb_checked"] = lab.efrsb_checked.fillna("0").astype(int)
LK = lab[["inn", "final_reason", "default_date", "stop_type", "clean", "efrsb_checked", "egrul_reason"]]

# ---------------- ordinary ----------------
u = pd.read_parquet(f"{SP}/buyers/universe_store_2024.parquet").drop(columns=["final_reason", "default_date", "stop_type", "egrul_reason", "efrsb_checked"], errors="ignore")
u = u.merge(LK, on="inn", how="left")
u["clean"] = u.clean.fillna(False).astype(bool)
u["rev_mln"] = u.line_2110 / 1000
base = u[u.clean & (u.age >= 1) & (u.eligible == 1) & (u.outlier == 0) & (u.imputed == 0)].copy()
base["level_"] = base.level
base["size"] = size_band(base.rev_mln); base["sect"] = sector_group(base.okved_section)
base["reg"] = region_group(base.region); base["ageb"] = age_band(base.age)
lvl_n = {k: v * N_BLOCKS for k, v in ORD_PER_LEVEL.items()}
tot = sum(lvl_n.values())
targets = {
    "level_": {k: v / tot for k, v in lvl_n.items()},
    "size": {"120-300": .30, "300-1000": .35, "1000-5000": .25, "5000+": .10},
    "sect": {"G": .22, "C": .20, "F": .15, "M": .10, "H": .08, "J": .05, "L": .04, "A": .04, "other": .12},
    "reg": {"moscow_city": .20, "moscow_region": .09, "spb": .07, "other": .64},
    "ageb": {"1-2": .12, "3-9": .38, "10-19": .35, "20+": .15},
}
base["w"] = ipf(base, targets)
ord_parts = []
for lv, n in lvl_n.items():
    g = base[base.level_ == lv]
    p = (g.w / g.w.sum()).values
    ord_parts.append(draw(g, n, p))
ordd = pd.concat(ord_parts)
# blocks: per level, order by (size, sector, random) and deal cards
ordd["rk"] = rng.random(len(ordd))
ordd = ordd.sort_values(["level_", "size", "sect", "rk"])
ordd["block"] = ordd.groupby("level_").cumcount() % N_BLOCKS + 1
ordd["category"] = "обычный"; ordd["sub_category"] = ""

# ---------------- stop-factor ----------------
STOP_REASONS = {"банкротство": ["банкротство"], "ликвидация": ["добровольная"], "исключение": ["исключение"]}
stop_parts = []
for t, reasons in STOP_REASONS.items():
    c = u[(u.stop_type == t) & u.final_reason.isin(reasons) & u.default_date.between(WIN_START, D_END) & (u.age >= 1)
          & (u.eligible == 1) & (u.outlier == 0)].copy()
    c["sect"] = sector_group(c.okved_section)
    c["lvl_flat"] = c.level
    tg = {"lvl_flat": flat(c.level, .5), "sect": flat(c.sect, .5)}
    p = ipf(c, tg)
    s = draw(c, STOP_PER_TYPE * N_BLOCKS, p).copy()
    s["block"] = rng.permutation(len(s)) % N_BLOCKS + 1
    s["category"] = "стоп_фактор"; s["sub_category"] = t
    stop_parts.append(s)
stop = pd.concat(stop_parts)

# ---------------- 'мало данных' ----------------
lost = pd.read_parquet(f"{UP}/lost_buyers_2024.parquet").merge(LK, on="inn", how="left")
lost["clean"] = lost.clean.fillna(False).astype(bool)
c23 = pd.read_parquet(f"{UP}/cohort_2023_raw_features.parquet", columns=["inn", "line_2110"]).rename(columns={"line_2110": "rev23"})
lost = lost.merge(c23, on="inn", how="left")
lost = lost[lost.dissolution_date.isna() & lost.clean & ~lost.okved_section.isin(["K", "O"]) & lost.okved_section.notna()]


def pick(df, n_per_block, label):
    d = df.copy()
    d["sect"] = sector_group(d.okved_section); d["reg"] = region_group(d.region)
    p = ipf(d, {"sect": flat(d.sect, .5), "reg": {"moscow_city": .2, "moscow_region": .09, "spb": .07, "other": .64}})
    s = draw(d, n_per_block * N_BLOCKS, p).copy()
    s["block"] = rng.permutation(len(s)) % N_BLOCKS + 1
    s["category"] = "мало_данных"; s["sub_category"] = label
    return s


md_a = pick(lost[(lost.filed == 0) & (lost.age >= 1)], MD_PER_BLOCK["no_report"], "no_report")
md_b = pick(lost[(lost.filed == 1) & lost.line_2110.between(20000, 119999) & (lost.age >= 1) & (lost.outlier != 1)], MD_PER_BLOCK["revenue_below_120m"], "revenue_below_120m")
ex = pd.read_parquet(f"{UP}/buyer_extras_2024.parquet")
young = pd.concat([
    u[(u.age < 1) & (u.eligible == 1)][["inn", "okved_section", "region", "age", "line_2110"]],
    ex[(ex.group == "young") & (ex.line_2110 >= 20000) & ex.okved_section.notna() & ~ex.okved_section.isin(["K", "O"])][["inn", "okved_section", "region", "age", "line_2110"]],
]).drop_duplicates("inn")
md_c = pick(young, MD_PER_BLOCK["age_below_1"], "age_below_1")
mal = pd.concat([md_a, md_b, md_c])
mal["rev_mln"] = np.where(mal.sub_category == "no_report", np.nan, mal.line_2110 / 1000)

# ---------------- 'нет данных' ----------------
nd = ex[(ex.group == "not_filed") & ex.dissolution_date.isna() & ex.okved_section.notna() & ~ex.okved_section.isin(["K", "O"])
        & (pd.to_datetime(ex.creation_date) <= "2022-12-31")].copy()
nd["sect"] = sector_group(nd.okved_section)
s = draw(nd, NO_DATA_PER_BLOCK * N_BLOCKS, ipf(nd, {"sect": flat(nd.sect, .5)})).copy()
s["block"] = rng.permutation(len(s)) % N_BLOCKS + 1
s["category"] = "нет_данных"; s["sub_category"] = ""
nodata = s

# ---------------- assemble ----------------
allsel = pd.concat([ordd, stop, mal, nodata], ignore_index=True)
assert allsel.inn.is_unique, "duplicate INN in pool"
rat = raw_ratios(pd.concat([ordd, stop]).reset_index(drop=True))
fin = pd.concat([ordd, stop]).reset_index(drop=True)[["inn"]].join(rat[["cash_ratio", "current_ratio", "payables_days"]].rename(columns=lambda c: "fx_" + c))
allsel = allsel.merge(fin, on="inn", how="left")
has_pd = allsel.category.isin(["обычный", "стоп_фактор"])
pool = pd.DataFrame({
    "buyer_inn": allsel.inn, "block": allsel.block, "category": allsel.category, "sub_category": allsel.sub_category,
    "stop_type": np.where(allsel.category == "стоп_фактор", allsel.sub_category, ""),
    "snapshot_date": D_END.date().isoformat(),
    "okved_section": np.where(allsel.category == "нет_данных", None, allsel.okved_section),
    "region": np.where(allsel.category == "нет_данных", None, allsel.region),
    "age_years": np.where(allsel.category == "нет_данных", np.nan, allsel.age),
    "revenue_mln": np.where(allsel.category.isin(["обычный", "стоп_фактор", "мало_данных"]), allsel.rev_mln, np.nan),
    "abs_liquidity": np.where(has_pd, allsel.fx_cash_ratio, np.nan),
    "current_liquidity": np.where(has_pd, allsel.fx_current_ratio, np.nan),
    "payables_days": np.where(has_pd, allsel.fx_payables_days, np.nan),
    "report_year": np.where(allsel.category.isin(["обычный", "стоп_фактор"]), 2024, np.where(allsel.sub_category.isin(["revenue_below_120m", "age_below_1"]), 2024, np.nan)),
    "pd": np.where(has_pd, allsel.pd, np.nan),
    "risk_level": np.where(has_pd, allsel.level, None),
    "in_buyers_csv": (allsel.category != "нет_данных").astype(int),
})
pool["buyer_name"] = ""
pool = pool.sort_values(["block", "category", "sub_category", "buyer_inn"]).reset_index(drop=True)
pool.to_csv(f"{OUT}/buyers_pool.csv", index=False, encoding="utf-8-sig")

# ---------------- status file ----------------
labs = LK.set_index("inn")
rows = []
for r in allsel.itertuples():
    if r.category == "нет_данных":
        continue
    in_lab = r.inn in labs.index
    chk = int(labs.efrsb_checked.get(r.inn, 0)) if in_lab else 0
    st = r.sub_category if r.category == "стоп_фактор" else ""
    egrul = {"": "действует", "банкротство": "действует", "ликвидация": "ликвидация", "исключение": "исключение"}[st]
    ev = labs.default_date.get(r.inn) if (st and in_lab) else pd.NaT
    rows.append(dict(buyer_inn=r.inn, snapshot_date=D_END.date().isoformat(), egrul_status=egrul,
                     bankruptcy_efrsb=int(st == "банкротство"), event_date=(ev.date().isoformat() if pd.notna(ev) else ""),
                     basis=("labels_egrul+efrsb" if (in_lab and chk) else ("labels_egrul_only" if in_lab else "no_labels")),
                     verified=int(in_lab and (chk == 1 or st == "банкротство"))))
status = pd.DataFrame(rows)
status.to_csv(f"{OUT}/buyers_status.csv", index=False, encoding="utf-8-sig")
status[status.verified == 0][["buyer_inn", "basis"]].to_csv(f"{OUT}/buyers_to_verify.csv", index=False)

# ---------------- diagnostics ----------------
rep = {"d_end": str(D_END.date()), "seed": SEED, "pool": len(pool),
       "by_category": pool.category.value_counts().to_dict(), "by_block": pool.block.value_counts().sort_index().to_dict(),
       "ordinary_levels_per_block": pd.crosstab(pool[pool.category == "обычный"].block, pool[pool.category == "обычный"].risk_level).to_dict(),
       "to_verify": int((status.verified == 0).sum()), "verify_basis": status[status.verified == 0].basis.value_counts().to_dict()}
json.dump(rep, open(f"{OUT}/pool_report.json", "w"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
