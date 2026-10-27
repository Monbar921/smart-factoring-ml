import pandas as pd, numpy as np, json, time
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pd_data import *
from pd_features import CAT_COLS
a, b, fb, Xa, Xb = load()
m = a.y_B_12m.notna().values
X, y = Xa[m].reset_index(drop=True), a.y_B_12m[m].astype(int).reset_index(drop=True)
for c in CAT_COLS: X[c] = X[c].astype(str)
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
rows = []
for depth, l2 in ((6, 10), (4, 10)):
    t = time.time(); aucs, its = [], []
    for tr, va in folds:
        mdl = CatBoostClassifier(iterations=2000, learning_rate=0.08, depth=depth, l2_leaf_reg=l2, loss_function="Logloss",
                                 eval_metric="AUC", auto_class_weights="SqrtBalanced", od_type="Iter", od_wait=100,
                                 random_seed=SEED, thread_count=2, verbose=False, cat_features=CAT_COLS)
        mdl.fit(X.iloc[tr], y.iloc[tr], eval_set=(X.iloc[va], y.iloc[va]), use_best_model=True)
        aucs.append(roc_auc_score(y.iloc[va], mdl.predict_proba(X.iloc[va])[:, 1])); its.append(mdl.get_best_iteration())
    r = dict(depth=depth, l2=l2, mean_gini=2 * np.mean(aucs) - 1, std_gini=2 * np.std(aucs), iters=int(np.mean(its)), sec=round(time.time() - t))
    rows.append(r); print(r, flush=True)
res = pd.DataFrame(rows).sort_values("mean_gini", ascending=False)
res.to_csv(OUT + "catboost_search.csv", index=False)
json.dump(res.iloc[0].to_dict(), open(OUT + "catboost_best.json", "w"), indent=1, default=float)
print("BEST", res.iloc[0].to_dict())
