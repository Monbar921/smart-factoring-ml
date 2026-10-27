import pandas as pd, numpy as np, json, time, itertools, random
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from pd_data import *
a, b, fb, Xa, Xb = load()
m = a.y_B_12m.notna().values
X, y = Xa[m].reset_index(drop=True), a.y_B_12m[m].astype(int).reset_index(drop=True)
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
rng = random.Random(SEED)
grid = dict(num_leaves=[7, 15, 31, 63], min_child_samples=[50, 100, 300, 600], feature_fraction=[0.5, 0.7, 0.9],
            lambda_l2=[1, 10, 50], pos_weight=[1, 5, 20])
cands = [dict(num_leaves=31, min_child_samples=100, feature_fraction=0.8, lambda_l2=10, pos_weight=1)]
while len(cands) < 14:
    c = {k: rng.choice(v) for k, v in grid.items()}
    if c not in cands: cands.append(c)
res = []
for i, c in enumerate(cands):
    t = time.time()
    p = dict(objective="binary", learning_rate=0.05, num_leaves=c["num_leaves"], min_child_samples=c["min_child_samples"],
             feature_fraction=c["feature_fraction"], bagging_fraction=0.8, bagging_freq=1, lambda_l2=c["lambda_l2"],
             metric="auc", verbose=-1, num_threads=2, seed=SEED)
    aucs, its = [], []
    for tr, va in folds:
        w = np.where(y.iloc[tr] == 1, c["pos_weight"], 1.0)
        mdl = lgb.train(p, lgb.Dataset(X.iloc[tr], y.iloc[tr], weight=w), 3000,
                        valid_sets=[lgb.Dataset(X.iloc[va], y.iloc[va])],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
        aucs.append(mdl.best_score["valid_0"]["auc"]); its.append(mdl.best_iteration)
    r = dict(c, mean_gini=2 * np.mean(aucs) - 1, std_gini=2 * np.std(aucs), iters=int(np.mean(its)), sec=round(time.time() - t))
    res.append(r); print(i, r, flush=True)
res = pd.DataFrame(res).sort_values("mean_gini", ascending=False)
res.to_csv(OUT + "lgbm_search.csv", index=False)
best = res.iloc[0].to_dict()
json.dump(best, open(OUT + "lgbm_best.json", "w"), indent=1, default=float)
print("BEST", best)
