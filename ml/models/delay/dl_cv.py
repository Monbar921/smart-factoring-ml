"""Подбор параметров: кросс-валидация группами по УПД внутри обучающего периода (проверочный период не используется)."""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import GroupKFold
import dl_common as dc

BASE = dict(objective="binary", learning_rate=0.04, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=10.0, verbose=-1, n_jobs=2, seed=7)


def folds(df, n=5):
    g = pd.factorize(df.upd_id)[0]
    return list(GroupKFold(n_splits=n).split(df, groups=g))


def cv_run(df, feats, k, params, n_splits=5, max_rounds=1200, log=print):
    y = df["y%d" % k].values
    ok = ~np.isnan(y)
    df, y = df[ok].reset_index(drop=True), y[ok]
    cats = [c for c in dc.CATEGORICAL if c in feats]
    oof = np.zeros(len(df))
    its = []
    for tr, va in folds(df, n_splits):
        dtr = lgb.Dataset(df.loc[tr, feats], y[tr], categorical_feature=cats, free_raw_data=False)
        dva = lgb.Dataset(df.loc[va, feats], y[va], categorical_feature=cats, reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=max_rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(60, verbose=False)])
        oof[va] = m.predict(df.loc[va, feats], num_iteration=m.best_iteration)
        its.append(m.best_iteration)
    s = (df.j.values == 0)
    res = dict(iters=int(np.mean(its)), logloss=log_loss(y, oof), auc_all=roc_auc_score(y, oof), auc_sign=roc_auc_score(y[s], oof[s]),
               auc_naive_sign=roc_auc_score(y[s], np.nan_to_num(df["naive_p%d" % k].values[s])),
               auc_ceiling_sign=roc_auc_score(y[s], df["p_true_gt_%d" % k].values[s]))
    return res, oof, df
