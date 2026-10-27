"""Оценки модели задержки на D_end для открытых УПД категорий 1–4 по всем десяти партиям (читаем CSV, как продукт)."""
import os
import sys
import time

import numpy as np
import pandas as pd

import dl_common as dc
import score_delay as sd

D_END = "2025-12-30"


def main(data_dir, out_path):
    d_end = int(dc.day_int(pd.Series([D_END]))[0])
    model = sd.get_model()
    parts = []
    for k in range(1, 11):
        t0 = time.time()
        tab = sd.load_tables(os.path.join(data_dir, "batch_%02d" % k))
        U, A = dc.reconcile(tab)
        ids = U.upd_id[np.isnan(U.close_d.values) & (U.plan_d.values >= d_end)].tolist()
        out = sd.score_delay(ids, D_END, tab, model)
        out.insert(0, "batch", k)
        parts.append(out.sort_values("upd_id"))
        print("batch_%02d: открытых УПД с категориями 1–4: %d, no_history: %d, %.1f с" % (k, len(out), (out.status == "no_history").sum(), time.time() - t0), flush=True)
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(out_path)
    print(df.groupby("batch")[["p_7", "p_14", "p_30", "p_90", "expected_days"]].mean().round(4))


if __name__ == "__main__":
    main(*sys.argv[1:])
