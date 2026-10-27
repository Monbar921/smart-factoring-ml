"""
Step 1. Cut cohorts 2022/2023 out of the company-year showcase (Vladimir's RFSD showcase),
check Gate 0 (showcase reproduces our INN lists), drop all post-cutoff columns,
add line_2120 from raw RFSD (year T only).
Usage: python3 01_extract_cohorts.py   (run from $HOME, absolute paths)
"""
import os, sys, json
import pyarrow.parquet as pq, pyarrow as pa, pandas as pd, numpy as np

B = os.path.expanduser("~/mnt/Downloads")
SHOW = f"{B}/pd_start/rfsd/company_year_strict_pit.parquet"
RAW = lambda y: f"{B}/rfsd/RFSD/year={y}/part-0.parquet"
INNS = {2022: f"{B}/pd_start/kogorta_2022_vse_inn.csv", 2023: f"{B}/pd_start/kogorta_2023_vse_inn.csv"}
OUT = f"{B}/pd_start/pd_work"

# columns that carry information after the cutoff date -> never leave the showcase
LEAK = ["dissolution_date", "cutoff", "label_end", "legal_default_12m", "legal_default_status",
        "legal_default_event_type", "legal_default_event_date", "label_source",
        "registered_cessation_12m", "registered_cessation_status", "registered_cessation_event_date",
        "observation_complete", "censor_reason"]
schema = pq.ParquetFile(SHOW).schema_arrow.names
keep = [c for c in schema if c not in LEAK]
print("showcase columns:", len(schema), "kept:", len(keep), "dropped:", [c for c in schema if c in LEAK])

parts = []
pf = pq.ParquetFile(SHOW)
for i in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(i, columns=keep).to_pandas()
    t = t[t.financial_year.isin([2022, 2023])]
    parts.append(t)
sh = pd.concat(parts, ignore_index=True)
del parts
print("showcase 2022/2023 rows:", len(sh))

report = {}
for y in (2022, 2023):
    inns = pd.read_csv(INNS[y], dtype=str)["inn"]
    d = sh[sh.financial_year == y].copy()
    report[f"rows_year_{y}"] = int(len(d))
    report[f"dup_inn_{y}"] = int(d.inn.duplicated().sum())
    # filters per TZ section 3
    f = d[(d.filed == 1) & (d.outlier != 1) & (~d.okved_section.isin(["K", "O"])) & (d.line_2110 >= 120000)]
    report[f"filtered_{y}"] = int(len(f))
    a, b = set(f.inn), set(inns)
    report[f"gate0_{y}"] = {"cohort_list": len(b), "showcase_filtered": len(a), "common": len(a & b),
                            "only_showcase": len(a - b), "only_list": len(b - a)}
    d = d[d.inn.isin(b)].copy()
    # line_2120 from raw RFSD
    raw = pq.read_table(RAW(y), columns=["inn", "line_2120"]).to_pandas()
    raw = raw[raw.inn.isin(b)]
    report[f"raw_dup_inn_{y}"] = int(raw.inn.duplicated().sum())
    raw = raw.drop_duplicates("inn")
    d = d.merge(raw, on="inn", how="left", validate="1:1")
    report[f"line_2120_notnull_{y}"] = int(d.line_2120.notna().sum())
    report[f"rows_out_{y}"] = int(len(d))
    d.to_parquet(f"{OUT}/cohort_{y}_raw_features.parquet", index=False)
    print(y, json.dumps({k: v for k, v in report.items() if str(y) in k}, ensure_ascii=False))
json.dump(report, open(f"{OUT}/01_extract_report.json", "w"), ensure_ascii=False, indent=1)
