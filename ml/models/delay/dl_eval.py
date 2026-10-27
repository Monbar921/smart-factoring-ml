"""Проверка модели задержки на проверочном периоде (месяцы 25–29 окна) по критериям раздела 8.5 ТЗ."""
import json
import os
import sys
import time

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dl_common as dc
import dl_model as dm
import score_delay as sd

ART = "art"
KS = dm.KS


def ece(y, p, bins=10):
    order = np.argsort(p, kind="stable")
    ch = np.array_split(order, bins)
    return float(sum(len(c) / len(y) * abs(y[c].mean() - p[c].mean()) for c in ch))


def deciles(y, p, bins=10):
    order = np.argsort(p, kind="stable")
    return [(float(p[c].mean()), float(y[c].mean()), int(len(c))) for c in np.array_split(order, bins)]


def auc(y, s):
    return float(roc_auc_score(y, s))


def main(data_dir="data"):
    t00 = time.time()
    ds = pd.read_parquet("ds_all.parquet")
    model = dm.DelayModel.load("delay_model_1_0.pkl")
    te = ds[(ds.part == "test") & (ds.no_history == 0)].reset_index(drop=True)
    P = model.predict(te)
    R = model.raw(te)
    sign = (te.j == 0).values
    res = dict(rows_test=int(len(te)), rows_sign=int(sign.sum()), upd_test=int(te.upd_id.nunique()))
    crit = []   # (проверка, требование, значение, прошла)

    # ---- качество p_k
    auc_t = {}
    for k in KS:
        lab = ~te["y%d" % k].isna().values
        s = sign & lab
        y = te["y%d" % k].values
        a = dict(model=auc(y[s], P["p_%d" % k].values[s]), model_raw=auc(y[s], R[k][s]),
                 naive=auc(y[s], te["naive_p%d" % k].fillna(0).values[s]),
                 ceiling=auc(y[s], te["p_true_gt_%d" % k].values[s]),
                 model_all_rows=auc(y[lab], P["p_%d" % k].values[lab]), n_sign=int(s.sum()), pos_rate=float(y[s].mean()))
        auc_t[k] = a
    res["auc"] = auc_t
    a14, a90 = auc_t[14], auc_t[90]
    crit.append(("AUC p_14 не ниже потолка минус 0,08", "≥ %.3f" % (a14["ceiling"] - 0.08), a14["model"], a14["model"] >= a14["ceiling"] - 0.08))
    crit.append(("AUC p_14 выше наивного правила не меньше чем на 0,03", "≥ %.3f" % (a14["naive"] + 0.03), a14["model"], a14["model"] >= a14["naive"] + 0.03))
    crit.append(("AUC p_90 не ниже потолка минус 0,10", "≥ %.3f" % (a90["ceiling"] - 0.10), a90["model"], a90["model"] >= a90["ceiling"] - 0.10))

    # ---- калибровка
    cal = {}
    for k in KS:
        y = te["y%d" % k].values
        lab = ~np.isnan(y)
        s = sign & lab
        cal[k] = dict(ece_sign=ece(y[s], P["p_%d" % k].values[s]), ece_all=ece(y[lab], P["p_%d" % k].values[lab]),
                      ece_raw_sign=ece(y[s], R[k][s]), deciles_sign=deciles(y[s], P["p_%d" % k].values[s]),
                      deciles_all=deciles(y[lab], P["p_%d" % k].values[lab]),
                      mean_pred_sign=float(P["p_%d" % k].values[s].mean()), mean_obs_sign=float(y[s].mean()))
        crit.append(("ECE p_%d (строки подписания)" % k, "≤ 0,03", cal[k]["ece_sign"], cal[k]["ece_sign"] <= 0.03))
    res["calibration"] = cal

    # ---- монотонность
    d = te[sign].copy()
    d["p14"] = P["p_14"].values[sign]
    lev = d[d.level.notna()].groupby("level").agg(p14=("p14", "mean"), obs=("y14", "mean"), n=("p14", "size"))
    lev.index = ["A", "B", "C", "D"][: len(lev)] if len(lev) == 4 else lev.index
    d["aq"] = pd.qcut(d.amount_ratio, 5, labels=False, duplicates="drop")
    amt = d[d.aq.notna()].groupby("aq").agg(p14=("p14", "mean"), obs=("y14", "mean"), ratio=("amount_ratio", "mean"), n=("p14", "size"))
    res["monotony"] = dict(level=lev.reset_index().to_dict("records"), amount=amt.reset_index().to_dict("records"))
    mon_lev = bool(np.all(np.diff(lev.p14.values) > 0))
    mon_amt = bool(np.all(np.diff(amt.p14.values) > -0.002) and amt.p14.values[-1] > amt.p14.values[0])
    crit.append(("Средняя p_14 растёт от уровня A к D", "A < B < C < D", " / ".join("%.3f" % v for v in lev.p14.values), mon_lev))
    crit.append(("Средняя p_14 растёт с ростом суммы к обычной для пары", "по пяти группам не убывает",
                 " / ".join("%.3f" % v for v in amt.p14.values), mon_amt))

    # ---- ожидаемые дни
    cl = te.delay_obs.notna().values
    pe = P.expected_days.values
    y = te.delay_obs.values
    nd = te.naive_days.values
    e = dict(n_closed_rows=int(cl.sum()), mae_model=float(np.mean(np.abs(pe[cl] - y[cl]))),
             mae_pair_median=float(np.mean(np.abs(nd[cl] - y[cl]))), mae_pair_mean=float(np.mean(np.abs(te.pair_mean.fillna(0).values[cl] - y[cl]))),
             mae_zero=float(np.mean(np.abs(y[cl]))), bias_model=float(np.mean(pe[cl] - y[cl])), mean_actual=float(y[cl].mean()),
             rmse_model=float(np.sqrt(np.mean((pe[cl] - y[cl]) ** 2))),
             rmse_pair_mean=float(np.sqrt(np.mean((te.pair_mean.fillna(0).values[cl] - y[cl]) ** 2))))
    s = sign & cl
    e["sign_mae_model"] = float(np.mean(np.abs(pe[s] - y[s])))
    e["sign_mae_pair_median"] = float(np.mean(np.abs(nd[s] - y[s])))
    e["sign_mae_oracle_mean"] = float(np.mean(np.abs(te.expected_delay_true.values[s] - y[s])))
    e["sign_bias_model"] = float(np.mean(pe[s] - y[s]))
    pm = te.pair_mean.fillna(0).values
    e["sign_mae_zero"] = float(np.mean(np.abs(y[s])))
    e["sign_mae_pair_mean"] = float(np.mean(np.abs(pm[s] - y[s])))
    e["sign_rmse_model"] = float(np.sqrt(np.mean((pe[s] - y[s]) ** 2)))
    e["sign_rmse_pair_median"] = float(np.sqrt(np.mean((nd[s] - y[s]) ** 2)))
    e["sign_rmse_pair_mean"] = float(np.sqrt(np.mean((pm[s] - y[s]) ** 2)))
    e["sign_rmse_oracle_mean"] = float(np.sqrt(np.mean((te.expected_delay_true.values[s] - y[s]) ** 2)))
    e["sign_rmse_zero"] = float(np.sqrt(np.mean(y[s] ** 2)))
    e["sign_rmse_model_vs_oracle"] = float(np.sqrt(np.mean((pe[sign] - te.expected_delay_true.values[sign]) ** 2)))
    e["sign_corr_model_oracle"] = float(np.corrcoef(pe[sign], te.expected_delay_true.values[sign])[0, 1])
    # смещение из-за выбора закрытых УПД: по правде, в которой исходы есть у всех
    for part in ("train", "test"):
        dd = ds[(ds.part == part) & (ds.no_history == 0) & (ds.j == 0)]
        e["selection_%s" % part] = dict(mean_true_all=float(dd.delay_true.mean()), mean_true_closed=float(dd[dd.delay_obs.notna()].delay_true.mean()),
                                        share_open_at_d_end=float(dd.delay_obs.isna().mean()))
    e["sign_bias_vs_truth_all"] = float(np.mean(pe[sign] - te.delay_true.values[sign]))
    res["expected_days"] = e
    crit.append(("MAE expected_days на закрытых УПД не хуже медианы по паре", "≤ %.2f" % e["sign_mae_pair_median"], e["sign_mae_model"],
                 e["sign_mae_model"] <= e["sign_mae_pair_median"]))
    crit.append(("Среднее смещение expected_days на закрытых УПД по модулю", "≤ 1 дня", e["sign_bias_model"], abs(e["sign_bias_model"]) <= 1.0))

    # ---- утечка
    for k in KS:
        crit.append(("AUC p_%d не выше потолка больше чем на 0,01" % k, "≤ %.3f" % (auc_t[k]["ceiling"] + 0.01), auc_t[k]["model"],
                     auc_t[k]["model"] <= auc_t[k]["ceiling"] + 0.01))
    rng = np.random.default_rng(11)
    tr = ds[(ds.part == "train") & (ds.no_history == 0) & ds.y14.notna()].reset_index(drop=True)
    uu = pd.Series(tr.groupby("upd_id").y14.first())
    sr = model.meta["search"][14]
    p = dict(sr["params"], verbose=-1, n_jobs=2)
    lab = sign & ~te.y14.isna().values
    a_sh_list = []
    for rep in range(5):
        perm = pd.Series(rng.permutation(uu.values), index=uu.index)
        m = lgb.train(p, lgb.Dataset(tr[model.features], tr.upd_id.map(perm).values, categorical_feature=model.categorical), sr["iters"])
        a_sh_list.append(auc(te.y14.values[lab], m.predict(te[model.features][lab])))
    a_sh = float(np.mean(a_sh_list))
    a_sh2 = float(np.mean([auc(rng.permutation(te.y14.values[lab]), P.p_14.values[lab]) for _ in range(5)]))
    res["shuffle"] = dict(train_shuffled=a_sh, train_shuffled_each=a_sh_list, test_labels_shuffled=a_sh2)
    crit.append(("AUC при перемешанных метках обучения (p_14), среднее из 5", "0,48–0,52", a_sh, 0.48 <= a_sh <= 0.52))
    crit.append(("AUC при перемешанных метках проверки (p_14), среднее из 5", "0,48–0,52", a_sh2, 0.48 <= a_sh2 <= 0.52))

    # ---- признаки не заглядывают вперёд: пересчёт по таблицам, урезанным до t
    samp = ds[ds.no_history == 0].sample(400, random_state=5)
    tabs, maxdiff, bad = {}, 0.0, 0
    for r in samp.itertuples():
        if r.batch not in tabs:
            tabs[r.batch] = dc.load_csv_batch(os.path.join(data_dir, "batch_%02d" % r.batch))
        U, rows, X = sd.feature_rows(tabs[r.batch], [r.upd_id], int(r.t), model.industries)
        a = X[model.features].values[0].astype(float)
        b = np.array([getattr(r, f) for f in model.features], dtype=float)
        same = np.isclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True)
        if not same.all():
            bad += 1
            maxdiff = max(maxdiff, float(np.nanmax(np.abs(a - b))))
    res["truncation_test"] = dict(rows=400, rows_with_diff=bad, max_abs_diff=maxdiff)
    crit.append(("Признаки при урезании таблиц до t совпадают (400 строк)", "0 расхождений", bad, bad == 0))

    # ---- скорость и повторяемость
    tab = sd.load_tables(os.path.join(data_dir, "batch_01"))
    U, A = dc.reconcile(tab)
    d_end = int(ds.t.max()) if False else dc.day_int(pd.Series(["2025-12-30"]))[0]
    open_ids = U.upd_id[(np.isnan(U.close_d.values)) & (U.plan_d.values >= d_end)].tolist()
    sd.score_delay(open_ids[:1], "2025-12-30", tab, model)      # прогрев: загрузка деревьев
    lat = []
    for u in open_ids[:30]:
        t0 = time.time()
        sd.score_delay([u], "2025-12-30", tab, model)
        lat.append(time.time() - t0)
    t0 = time.time()
    out1 = sd.score_delay(open_ids, "2025-12-30", tab, model)
    t_batch = time.time() - t0
    out2 = sd.score_delay(open_ids, "2025-12-30", tab, model)
    same = out1.fillna(-1).equals(out2.fillna(-1))
    t0 = time.time()
    tab2 = dc.load_csv_batch(os.path.join(data_dir, "batch_01"))
    t_load = time.time() - t0
    res["speed"] = dict(n_open=len(open_ids), single_mean=float(np.mean(lat)), single_max=float(np.max(lat)), batch_seconds=t_batch, load_seconds=t_load,
                        repeatable=bool(same))
    crit.append(("Один УПД", "≤ 1 с", float(np.max(lat)), float(np.max(lat)) <= 1.0))
    crit.append(("Пакет из %d УПД" % len(open_ids), "≤ 10 с", t_batch, t_batch <= 10.0))
    crit.append(("Повторяемость: тот же вход — тот же результат", "равны", int(same), bool(same)))
    res["criteria"] = [dict(name=a, requirement=b, value=(float(c) if not isinstance(c, str) else c), passed=bool(dd_)) for a, b, c, dd_ in crit]

    # ---- рисунки
    os.makedirs(ART + "/figs", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for a_, k in zip(ax, (14, 90)):
        lab = sign & ~te["y%d" % k].isna().values
        yk = te["y%d" % k].values[lab]
        for nm, sc, st in (("модель", P["p_%d" % k].values[lab], "-"), ("наивное правило", te["naive_p%d" % k].fillna(0).values[lab], "--"),
                           ("потолок (p_true)", te["p_true_gt_%d" % k].values[lab], ":")):
            fpr, tpr, _ = roc_curve(yk, sc)
            a_.plot(fpr, tpr, st, label="%s, AUC %.3f" % (nm, auc(yk, sc)))
        a_.plot([0, 1], [0, 1], color="gray", lw=0.6)
        a_.set_title("ROC, задержка больше %d дн. (строки подписания, проверка)" % k, fontsize=9)
        a_.set_xlabel("доля ложных тревог")
        a_.set_ylabel("доля найденных")
        a_.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(ART + "/figs/roc.png", dpi=130)
    plt.close()
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.8))
    for a_, k in zip(ax, KS):
        dc_ = cal[k]["deciles_sign"]
        xs, ys = [x[0] for x in dc_], [x[1] for x in dc_]
        mx = max(max(xs), max(ys)) * 1.05
        a_.plot([0, mx], [0, mx], color="gray", lw=0.6)
        a_.plot(xs, ys, "o-")
        a_.set_title("p_%d: ECE %.3f" % (k, cal[k]["ece_sign"]), fontsize=9)
        a_.set_xlabel("средняя оценка в дециле")
        a_.set_ylabel("доля задержек в дециле")
    plt.tight_layout()
    plt.savefig(ART + "/figs/calibration.png", dpi=130)
    plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a_, k in zip(ax, (14, 90)):
        b_ = lgb.Booster(model_str=model.boosters[k])
        imp = pd.Series(b_.feature_importance("gain"), model.features).sort_values()
        imp = (imp / imp.sum()).tail(15)
        a_.barh(imp.index, imp.values)
        a_.set_title("Важность признаков (gain), p_%d" % k, fontsize=9)
    plt.tight_layout()
    plt.savefig(ART + "/figs/importance.png", dpi=130)
    plt.close()
    res["importance"] = {k: dict(sorted(zip(model.features, (lambda g: (g / g.sum()).tolist())(np.array(lgb.Booster(model_str=model.boosters[k]).feature_importance("gain")))), key=lambda x: -x[1])[:15]) for k in KS}
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    o = np.argsort(pe[cl])
    ch = np.array_split(o, 10)
    ax[0].plot([pe[cl][c].mean() for c in ch], [y[cl][c].mean() for c in ch], "o-")
    mx = max(pe[cl].max(), 1)
    ax[0].plot([0, 30], [0, 30], color="gray", lw=0.6)
    ax[0].set_xlabel("ожидаемые дни по модели (среднее в дециле)")
    ax[0].set_ylabel("фактическая задержка закрытых УПД")
    ax[0].set_title("expected_days: оценка и факт по децилям", fontsize=9)
    ax[1].hist(y[cl], bins=np.arange(0, 100, 5), alpha=0.6, label="факт", density=True)
    ax[1].hist(pe[cl], bins=np.arange(0, 100, 5), alpha=0.6, label="модель", density=True)
    ax[1].legend()
    ax[1].set_title("Распределение дней задержки", fontsize=9)
    plt.tight_layout()
    plt.savefig(ART + "/figs/expected_days.png", dpi=130)
    plt.close()
    res["elapsed_s"] = time.time() - t00
    json.dump(res, open(ART + "/eval_results.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    for c in res["criteria"]:
        print(("OK  " if c["passed"] else "НЕТ ") + c["name"], "|", c["requirement"], "|", c["value"])


if __name__ == "__main__":
    main(*sys.argv[1:])
