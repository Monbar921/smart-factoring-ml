"""Обучение модели задержки: подбор параметров кросс-валидацией, OOF-калибровка, финальные модели. Проверочный период не трогаем."""
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

import dl_common as dc
import dl_model as dm
from dl_cv import BASE, cv_run, folds

ART = "art"
GRID = {"A": {}, "B": dict(num_leaves=15, min_child_samples=800, colsample_bytree=0.6, reg_lambda=50.0),
        "C": dict(num_leaves=7, min_child_samples=2000, colsample_bytree=0.6, reg_lambda=50.0)}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fit_fixed(dtr_df, y, feats, params, iters):
    cats = [c for c in dc.CATEGORICAL if c in feats]
    return lgb.train(params, lgb.Dataset(dtr_df[feats], y, categorical_feature=cats), num_boost_round=iters)


def oof_fixed(df, y, feats, params, iters):
    oof = np.zeros(len(df))
    for tr, va in folds(df, 5):
        m = fit_fixed(df.iloc[tr], y[tr], feats, params, iters)
        oof[va] = m.predict(df.iloc[va][feats])
    return oof


def choose_calibration(p, y, groups):
    """Platt, isotonic или без калибровки — по логарифмической потере на половинах OOF (перекрёстно)."""
    half = groups % 2
    res = {}
    for kind in ("none", "platt", "isotonic"):
        ll = []
        for h in (0, 1):
            cal = dm.Calibrator.fit(kind, p[half != h], y[half != h])
            ll.append(log_loss(y[half == h], np.clip(cal.apply(p[half == h]), 1e-6, 1 - 1e-6)))
        res[kind] = float(np.mean(ll))
    best = min(("platt", "isotonic"), key=res.get)     # по ТЗ калибровка — Platt или isotonic; «без калибровки» только справочно
    return best, res


def main():
    os.makedirs(ART, exist_ok=True)
    ds = pd.read_parquet("ds_all.parquet")
    meta = json.load(open("ds_all.parquet.json"))
    tr = ds[(ds.part == "train") & (ds.no_history == 0)].reset_index(drop=True)
    feats = dc.FEATURES
    log("обучающих строк", len(tr), "УПД", tr.upd_id.nunique())
    search = {}
    boosters, cals, cal_info, oof_store = {}, {}, {}, {}
    for k in dm.KS:
        y = tr["y%d" % k].values
        lab = ~np.isnan(y)
        d, yk = tr[lab].reset_index(drop=True), y[lab]
        best = None
        for name, upd in GRID.items():
            p = dict(BASE)
            p.update(upd)
            t0 = time.time()
            res, _, _ = cv_run(d, feats, k, p)
            log("k=%d конфиг %s: %s (%.0f с)" % (k, name, {a: round(v, 4) if isinstance(v, float) else v for a, v in res.items()}, time.time() - t0))
            if best is None or res["logloss"] < best[1]["logloss"]:
                best = (name, res, p)
        name, res, p = best
        iters = int(res["iters"])
        search[k] = dict(config=name, params={a: v for a, v in p.items() if a not in ("verbose", "n_jobs")}, iters=iters, cv=res)
        log("k=%d выбран %s, деревьев %d" % (k, name, iters))
        # OOF с фиксированным числом деревьев и калибровка на них
        oof = oof_fixed(d, yk, feats, p, iters)
        grp = pd.factorize(d.upd_id)[0]
        kind, ll = choose_calibration(oof, yk, grp)
        cals[k] = dm.Calibrator.fit(kind, oof, yk)
        cal_info[k] = dict(kind=kind, logloss=ll, oof_auc=float(roc_auc_score(yk, oof)))
        log("k=%d калибровка %s %s" % (k, kind, {a: round(v, 5) for a, v in ll.items()}))
        pd.DataFrame(dict(upd_id=d.upd_id, t=d.t, batch=d.batch, y=yk, oof=oof)).to_parquet("%s/oof_p%d.parquet" % (ART, k))
        m = fit_fixed(d, yk, feats, p, iters)
        boosters[k] = m.model_to_string()
    json.dump(dict(search=search, calibration=cal_info), open(ART + "/search.json", "w"), indent=1, default=float)
    # ---- ожидаемые дни: регрессия на закрытых УПД
    cl = tr[tr.delay_obs.notna()].reset_index(drop=True)
    y = cl.delay_obs.values
    regs = {"l2": dict(BASE, objective="regression", num_leaves=15, min_child_samples=800, colsample_bytree=0.6, reg_lambda=50.0),
            "tweedie": dict(BASE, objective="tweedie", tweedie_variance_power=1.3, num_leaves=15, min_child_samples=800,
                            colsample_bytree=0.6, reg_lambda=50.0)}
    reg_res = {}
    cats = dc.CATEGORICAL
    for name, p in regs.items():
        oof = np.zeros(len(cl))
        its = []
        for trn, va in folds(cl, 5):
            m = lgb.train(p, lgb.Dataset(cl.loc[trn, feats], y[trn], categorical_feature=cats), 1500,
                          valid_sets=[lgb.Dataset(cl.loc[va, feats], y[va], categorical_feature=cats)],
                          callbacks=[lgb.early_stopping(60, verbose=False)])
            oof[va] = np.maximum(m.predict(cl.loc[va, feats], num_iteration=m.best_iteration), 0)
            its.append(m.best_iteration)
        reg_res[name] = dict(iters=int(np.mean(its)), rmse=float(np.sqrt(np.mean((oof - y) ** 2))), mae=float(np.mean(np.abs(oof - y))),
                             bias=float(np.mean(oof - y)), mae_pair_median=float(np.mean(np.abs(np.nan_to_num(cl.naive_days.values) - y))),
                             mae_zero=float(np.mean(np.abs(y))), mean_y=float(y.mean()))
        log("expected_days", name, reg_res[name])
    bestr = min(reg_res, key=lambda n: reg_res[n]["rmse"])
    m = lgb.train(regs[bestr], lgb.Dataset(cl[feats], y, categorical_feature=cats), reg_res[bestr]["iters"])
    reg = dict(kind=bestr, model=m.model_to_string(), shift=0.0)
    json.dump(reg_res, open(ART + "/reg_search.json", "w"), indent=1)
    model = dm.DelayModel(feats, dc.CATEGORICAL, meta["industries"], boosters, cals, reg,
                          dict(trained_rows=int(len(tr)), bounds=meta["bounds"], search=search, calibration=cal_info, reg=reg_res))
    model.save("delay_model_1_0.pkl")
    log("модель сохранена")


if __name__ == "__main__":
    main()
