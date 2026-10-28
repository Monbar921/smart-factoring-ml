# Модель PD покупателя (pd-1.0)

Вероятность дефолта покупателя на 12 месяцев, уровень риска A–D и три главные причины. Один бустинг LightGBM, калибровка Платта. Обучена на отчётности RFSD за 2022 год, проверена на 2023. Результаты: `Smart_Factoring_PD_Model_Report_v1.0.pdf`. Суммы в модели в тысячах рублей.

Уровни по PD: A — до 0,4%, B — до 1,5%, C — до 6%, D — выше. PD ограничена диапазоном 0,05%–95%.

## Что нужно для запуска

Три файла в одной папке: `pd_model.pkl`, `pd_model.py`, `pd_features.py`. Имена модулей менять нельзя: в пикле лежат ссылки на классы `pd_features.FeatureBuilder` и `pd_model.PlattCalibrator`. Зависимости: `pandas`, `numpy`, `lightgbm`, `scikit-learn` (его требует `pd_model.py`). Проверка загрузки: `python -c "from pd_model import PDModel; PDModel.load('pd_model.pkl')"`. Остальные файлы нужны только для обучения и проверки.

## Подключение к сервису (ml-service-openapi.yaml)

Сервис ждёт модуль `app.models.pd_trained` с функцией `create()`: у модели `version`, `trained_at`, `stub = False` и метод `score(buyers, calc_date)`, который возвращает список `PdResult` в порядке запроса. Работающий образец лежит в `service_example.py` (проверен на выдуманном покупателе): он собирает вход модели из покупателя запроса и формирует ответ.

**Вход.** Для каждого покупателя нужна одна строка таблицы со столбцами `inn`, `okved_section`, `region`, `simplified`, `age` и столбцами `line_<код>`, `line_<код>_t1`, `line_<код>_t2` (отчётность за год T, T−1 и T−2). Столбцы должны быть все, пропущенное значение — `NaN`. Соответствие строк формы полям `financials`:

2110 — `revenue`, 2120 — `cost_of_sales` (только за T), 2200 — `sales_profit`, 2330 — `interest_payable`, 2400 — `net_profit`, 1200 — `current_assets`, 1230 — `receivables`, 1250 — `cash`, 1300 — `equity`, 1410 — `long_term_borrowings`, 1500 — `short_term_liabilities`, 1510 — `short_term_borrowings`, 1520 — `payables`, 1600 — `assets_total`.

Рубли из запроса делятся на 1000. Год T — последний год в `financials`. `age` = T минус год `registration_date`. `region` приводится к значениям RFSD: `moscow city`, `moscow reg.`, `sankt-petersburg`, всё остальное любое другое (важны только эти три). `simplified` — 0 или 1.

**Выход.** `X = model.features(raw)`, `p = model.predict_pd(X)`, `model.level(p)`, `model.top_factors(X, 3)`. В `PdResult`: `status = SCORED`, `pd_source = MODEL`, `low_data = false`, `pd = p`. Причины (`reasons`) берутся из `top_factors`:

- `feature` — как есть; `title` — `name_ru` **с заглавной буквы** («Доля собственного капитала»): название попадает в промпт для GigaChat и в текст менеджеру дословно;
- `direction` — `raises_risk` → `INCREASES_RISK`, `lowers_risk` → `DECREASES_RISK`; `contribution` — `shap`;
- `value` — число из `value` только для обычных показателей (доли, дни оборота, возраст, ликвидность). Для размера выручки и активов (`log_revenue`, `log_assets`) значение логарифмическое, для флагов (убыток, отрицательный капитал и т.п.) это 0/1, для отрасли и региона — текст. Для них `value` оставляй пустым (`null`): название причины и направление скажут всё сами, а число «15,8» вместо размера выручки менеджеру ничего не говорит.

**Когда модель не определена** (базовый PD по отрасли и размеру, `pd_source = BASE`, `low_data = true`): нет отчётности за последний год (`NO_RECENT_REPORT`), выручка ниже 120 млн ₽ (`LOW_REVENUE`), компания моложе года (`YOUNG_COMPANY`). Разделы ОКВЭД K и O модель тоже не знает. Базовый PD в этом пакете нет, берите из эвристики сервиса.

## Локальный запуск по ИНН (нужно хранилище отчётности)

```python
from score_inn import score, score_many
score("7707083893")            # PD, уровень A-D, три причины; status "no_financials", если ИНН нет в хранилище
score_many(["...", "..."])     # то же для списка, таблицей
```
Нужны `pd_model.pkl` и `data/scoring_store_2024.parquet` (пути переопределяются `PD_MODEL_PATH` / `PD_STORE_PATHS`). В сервисе хранилище не нужно: отчётность приходит в запросе.

## Порядок пересборки модели

1. `laptop_scripts/01_extract_cohorts.py`, `07_extract_scoring_store.py` — вырезают когорты и год для расчёта из витрины RFSD (запускать там, где она лежит). Результат — в `data/`.
2. В `data/` нужны ещё `cohort_2022_labels.csv`, `cohort_2023_labels.csv`.
3. `02_lgbm_search.py` → `03_logreg_cv.py` → `04_catboost_cv.py` → `05a_oof_calib.py` → `05b_evaluate.py 0.004 0.015 0.06` → `05c_compare_models.py` → `06_label_variants.py` → `08_build_tables.py` → `09_plots.py`; `verify_pd.py` перепроверяет главные цифры, `build_notebook.py` собирает ноутбук.
4. Если запускаете из другой папки, задайте `PD_ROOT`.

Версия RFSD: `ca3665997c977ba8ad01cf93c3e2a74ce7fe9c05`.
