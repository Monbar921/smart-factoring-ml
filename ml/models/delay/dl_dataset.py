"""Сборка выборки модели задержки: строки (УПД, дата t), метки, признаки, части по времени (раздел 8.2 ТЗ)."""
import json
import os
import sys

import numpy as np
import pandas as pd

import dl_common as dc

D_END_STR = "2025-12-30"


def part_bounds(d_end_str=D_END_STR, window_months=33):
    d_end = pd.Timestamp(d_end_str)
    start = d_end - pd.DateOffset(months=window_months)
    day = lambda m: int(dc.day_int(pd.Series([start + pd.DateOffset(months=m)]))[0])
    return dict(start=int(dc.day_int(pd.Series([start]))[0]), b3=day(3), b20=day(20), b24=day(24), b29=day(29),
                d_end=int(dc.day_int(pd.Series([d_end]))[0]))


def part_of(upd_d, b):
    return np.select([upd_d < b["b3"], upd_d < b["b20"], upd_d < b["b24"], upd_d < b["b29"]],
                     ["history", "train", "gap", "test"], default="rest")


def build_dataset(data_dir, truth_path, batches=range(1, 11), log=print):
    b = part_bounds()
    d_end = b["d_end"]
    tabs = {k: dc.load_csv_batch(os.path.join(data_dir, "batch_%02d" % k)) for k in batches}
    industries = sorted(set().union(*[set(t["cl"].industry) for t in tabs.values()]))
    truth = pd.read_parquet(truth_path)
    truth = truth.assign(delay_true=truth.delay_days).set_index("upd_id")[
        ["delay_true", "open_at_d_end", "unpaid_180", "p_true_gt_7", "p_true_gt_14", "p_true_gt_30", "p_true_gt_90",
         "expected_delay_true", "trend_flag"]]
    parts = []
    for k, tab in tabs.items():
        U, A = dc.reconcile(tab)
        part = part_of(U.upd_d.values, b)
        rows = dc.make_rows(U, d_end, mask=(part != "history"))
        X = dc.features(tab, U, A, rows, industries)
        lab = dc.make_labels(U, rows, d_end)
        nv = dc.baselines(X)
        meta = pd.DataFrame(dict(batch=k, upd_id=U.upd_id.values[rows.u.values], t=rows.t.values, j=rows.j.values,
                                 part=part[rows.u.values], plan_d=U.plan_d.values[rows.u.values],
                                 upd_d=U.upd_d.values[rows.u.values], buyer_inn=U.buyer_inn.values[rows.u.values],
                                 client_id=U.client_id.values[rows.u.values], pair=U.pair.values[rows.u.values]))
        df = pd.concat([meta, X, lab, nv], axis=1)
        df = df.join(truth, on="upd_id")
        parts.append(df)
        log("batch_%02d: УПД %d, строк %d, части: %s" % (k, len(U), len(df), df.part.value_counts().to_dict()))
    ds = pd.concat(parts, ignore_index=True)
    ds["no_history"] = ((ds.pair_n == 0) & (ds.buy_n == 0)).astype(int)
    return ds, industries, b


if __name__ == "__main__":
    ds, industries, b = build_dataset(sys.argv[1], sys.argv[2])
    out = sys.argv[3]
    ds.to_parquet(out)
    json.dump(dict(industries=industries, bounds=b), open(out + ".json", "w"))
    print(ds.shape)
    print(ds.groupby("part").agg(rows=("upd_id", "size"), upds=("upd_id", "nunique"), no_hist=("no_history", "sum")))
