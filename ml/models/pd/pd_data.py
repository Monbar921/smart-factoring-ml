"""Paths and loader. Data folder layout (relative to this file, or PD_ROOT):
   data/cohort_{2022,2023}_raw_features.parquet   - result of laptop_scripts/01_extract_cohorts.py
   data/cohort_{2022,2023}_labels.csv             - final labels (TZ v1.1)
   data/scoring_store_2024.parquet                - result of laptop_scripts/07_extract_scoring_store.py
   out/                                           - all results
"""
import os
import pandas as pd, numpy as np
from pd_features import FeatureBuilder, FEATURES
ROOT = os.environ.get("PD_ROOT", os.path.dirname(os.path.abspath(__file__)))
U = L = os.path.join(ROOT, "data") + os.sep
OUT = os.path.join(ROOT, "out") + os.sep
SEED = 42

def load():
    a = pd.read_parquet(U + "cohort_2022_raw_features.parquet").merge(
        pd.read_csv(L + "cohort_2022_labels.csv", dtype={"inn": str}, low_memory=False), on="inn", validate="1:1")
    b = pd.read_parquet(U + "cohort_2023_raw_features.parquet").merge(
        pd.read_csv(L + "cohort_2023_labels.csv", dtype={"inn": str}, low_memory=False), on="inn", validate="1:1")
    fb = FeatureBuilder().fit(a)          # clip bounds from the training cohort only
    return a, b, fb, fb.transform(a), fb.transform(b)

def gini(y, p):
    from sklearn.metrics import roc_auc_score
    return 2 * roc_auc_score(y, p) - 1
