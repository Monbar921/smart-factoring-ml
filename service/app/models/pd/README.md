# PD model package (v1.0)

Buyer PD model for «Смарт-факторинг». Trained on cohort 2022, checked on cohort 2023. Results: `Smart_Factoring_PD_Model_Report_v1.0.pdf`.

## Use
```python
from score_inn import score, score_many
score("7707083893")            # PD, level A-D, three factors; status "no_financials" if the company is not in the store
score_many(["...", "..."])     # same for a list, as a table
```
Needs `out/pd_model.pkl` and `data/scoring_store_2024.parquet` (set `PD_MODEL_PATH` / `PD_STORE_PATHS` to override). Python 3.10+, pandas, numpy, lightgbm, scikit-learn.

## Rebuild order
1. `laptop_scripts/01_extract_cohorts.py`, `07_extract_scoring_store.py` — cut cohorts and the scoring year out of the RFSD showcase (run on the machine with the showcase). Outputs go to `data/`.
2. `data/` must also hold `cohort_2022_labels.csv`, `cohort_2023_labels.csv`.
3. `02_lgbm_search.py` → `03_logreg_cv.py` → `04_catboost_cv.py` → `05a_oof_calib.py` → `05b_evaluate.py 0.004 0.015 0.06` → `05c_compare_models.py` → `06_label_variants.py` → `08_build_tables.py` → `09_plots.py`; `verify_pd.py` re-checks the headline numbers; `build_notebook.py` builds the notebook.
4. Set `PD_ROOT` to this folder if you run from elsewhere.

RFSD revision: `ca3665997c977ba8ad01cf93c3e2a74ce7fe9c05`. Amounts in thousand RUB.
