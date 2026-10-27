"""
Scoring store: company-year rows for the latest reporting year (2024) from the showcase,
same filters and same columns as the training cohorts (no labels, no post-cutoff columns).
Used by score(inn) to score real buyers.
"""
import os, json
import pyarrow.parquet as pq, pandas as pd

B = os.path.expanduser("~/mnt/Downloads")
SHOW = f"{B}/pd_start/rfsd/company_year_strict_pit.parquet"
RAW = lambda y: f"{B}/rfsd/RFSD/year={y}/part-0.parquet"
OUT = f"{B}/pd_start/pd_work"
YEAR = 2024
LEAK = ["dissolution_date", "cutoff", "label_end", "legal_default_12m", "legal_default_status",
        "legal_default_event_type", "legal_default_event_date", "label_source",
        "registered_cessation_12m", "registered_cessation_status", "registered_cessation_event_date",
        "observation_complete", "censor_reason"]
pf = pq.ParquetFile(SHOW)
keep = [c for c in pf.schema_arrow.names if c not in LEAK]
parts = []
for i in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(i, columns=keep).to_pandas()
    parts.append(t[t.financial_year == YEAR])
d = pd.concat(parts, ignore_index=True)
n_all = len(d)
d = d[(d.filed == 1) & (d.outlier != 1) & (~d.okved_section.isin(["K", "O"])) & (d.line_2110 >= 120000)].copy()
raw = pq.read_table(RAW(YEAR), columns=["inn", "line_2120"]).to_pandas().drop_duplicates("inn")
d = d.merge(raw, on="inn", how="left", validate="1:1")
d.to_parquet(f"{OUT}/scoring_store_{YEAR}.parquet", index=False)
rep = dict(year=YEAR, rows_year=n_all, rows_after_filters=len(d), dup_inn=int(d.inn.duplicated().sum()),
           revenue_ge_800m=int((d.line_2110 >= 800000).sum()), rfsd_revision=str(d.rfsd_revision.iloc[0]))
json.dump(rep, open(f"{OUT}/07_scoring_store_report.json", "w"), indent=1)
print(rep)
