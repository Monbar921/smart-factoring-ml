import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s)); code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# Модель PD покупателя: обучение и проверка
Версия 1.0 · «Смарт-факторинг» · по ТЗ «Обучение модели PD покупателя» v1.1

Что делает ноутбук: берёт выборку из витрины RFSD (когорты 2022 и 2023), строит признаки по разделу 6 ТЗ, обучает бустинг и логистическую регрессию на когорте 2022, калибрует PD, задаёт уровни A–D, проверяет на когорте 2023 и показывает главные факторы по SHAP. Подбор параметров (долгий) уже выполнен скриптами `02`–`04`, его результаты лежат в `out/`; здесь параметры берутся оттуда.

**Правило:** когорта 2023 в настройке не участвует: ни признаки, ни параметры, ни калибровка, ни границы уровней.""")
code("""import json, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from pd_data import *
from pd_features import FEATURES, NUM_COLS, RU_NAMES
from pd_model import *
from pd_metrics import *
pd.set_option("display.width", 160)
a, b, fb, Xa, Xb = load()
print("когорта 2022:", len(a), " когорта 2023:", len(b))""")
md("## 1. Гейт 0: витрина воспроизводит наши списки ИНН")
code("""import os
g = json.load(open(os.path.join(ROOT, "out", "01_extract_report.json")))
pd.DataFrame({k.replace("gate0_", "когорта "): v for k, v in g.items() if k.startswith("gate0")}).T""")
md("""## 2. Выборка и метки
Основная метка — `y_B_12m` (банкротство или исключение из ЕГРЮЛ в окне 12 месяцев после даты среза). Строки с пустой меткой в обучение и проверку не берутся.""")
code("""rows = []
for coh, d in (("2022 (обучение)", a), ("2023 (проверка)", b)):
    for c in ("y_B_12m", "y_A_12m", "y_0_12m", "y_B_21m"):
        if c in d.columns and d[c].notna().any():
            rows.append(dict(когорта=coh, метка=c, строк=int(d[c].notna().sum()), дефолтов=int(d[c].sum()), доля=round(float(d[c].mean()) * 100, 2)))
pd.DataFrame(rows)""")
md("## 3. Признаки\nФормулы и значения полей — в `out/pd_fields.md`. Коэффициенты обрезаны по 1-му и 99-му перцентилю когорты 2022, деление на ноль даёт пропуск.")
code("""print(len(FEATURES), "признаков:", FEATURES)
Xa[NUM_COLS].describe().T[["count", "mean", "min", "50%", "max"]].round(3)""")
md("## 4. Бустинг (LightGBM): кросс-валидация на 2022")
code("""best = json.load(open(OUT + "lgbm_best.json")); iters = int(best["iters"])
print({k: best[k] for k in ("num_leaves", "min_child_samples", "feature_fraction", "lambda_l2", "pos_weight", "iters")})
m = a.y_B_12m.notna().values
Xt, yt = Xa[m].reset_index(drop=True), a.y_B_12m[m].astype(int).values
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xt, yt))
oof_raw = np.zeros(len(yt))
for k, (tr, va) in enumerate(folds):
    mdl = fit_lgb(Xt.iloc[tr], yt[tr], best, iters); oof_raw[va] = mdl.predict(Xt.iloc[va])
    print("часть", k + 1, "Gini", round(gini(yt[va], oof_raw[va]), 4))
print("Gini вне обучающих частей:", round(gini(yt, oof_raw), 4))""")
md("Результаты подбора параметров (14 вариантов, пять частей):")
code("""pd.read_csv(OUT + "lgbm_search.csv").round(4)""")
md("## 5. Логистическая регрессия и CatBoost для сравнения\nВ кросс-валидации по 2022 и на проверке 2023 (таблица посчитана скриптом `05c`, на тех же частях):")
code("""pd.read_csv(OUT + "model_comparison.csv").round(4)""")
md("""Бустинг остаётся. Он лучше обычной логистической регрессии на 0,10 Gini в кросс-валидации (в каждой из пяти частей) и на 0,09 на проверке 2023. Регрессия со сплайнами (нелинейные члены) ближе: бустинг лучше неё в каждой из пяти частей, в среднем на 0,024, на проверке 2023 на 0,016. CatBoost слабее: на 0,007 в кросс-валидации и на 0,019 на проверке.""")
md("## 6. Калибровка\nОбучали с весом класса 5, поэтому «сырые» скоры завышены. Калибровку строим на предсказаниях вне обучающих частей. Сравнили изотоническую и Platt на перекрёстной проверке (логлосс, Brier):")
code("""print(json.dumps(json.load(open(OUT + "calibration_methods.json")), indent=1))
cal = fit_calibrator(oof_raw, yt)
oof_pd = np.clip(cal.predict(oof_raw), PD_FLOOR, PD_CAP)
print("Platt: a=%.3f b=%.3f, средний PD %.4f, доля дефолтов %.4f" % (cal.a, cal.b, oof_pd.mean(), yt.mean()))""")
md("## 7. Уровни риска A–D\nГраницы заданы по 2022 (вне обучающих частей) так, чтобы доля дефолтов росла от A к D: A < 0,4%, B 0,4–1,5%, C 1,5–6%, D ≥ 6%.")
code("""cuts = [0.004, 0.015, 0.06]
level_table(yt, assign_level(oof_pd, cuts)).round(4)""")
md("## 8. Финальная модель и проверка на когорте 2023")
code("""booster = fit_lgb(Xt, yt, best, int(round(iters * 1.1)))
model = PDModel(fb, booster, cal, cuts)
mb = b.y_B_12m.notna().values; m21 = b.y_B_21m.notna().values
y12, y21 = b.y_B_12m[mb].astype(int).values, b.y_B_21m[m21].astype(int).values
p12, p21 = model.predict_pd(Xb[mb]), model.predict_pd(Xb[m21])
seg = (b.loc[mb, "line_2110"] >= 800000).values
out = pd.DataFrame([
    dict(выборка="12 мес., все", n=len(y12), дефолтов=int(y12.sum()), Gini=gini(y12, p12), ci=boot_gini(y12, p12, n=300)),
    dict(выборка="12 мес., выручка от 800 млн ₽", n=int(seg.sum()), дефолтов=int(y12[seg].sum()), Gini=gini(y12[seg], p12[seg]), ci=boot_gini(y12[seg], p12[seg], n=300)),
    dict(выборка="21 мес., все", n=len(y21), дефолтов=int(y21.sum()), Gini=gini(y21, p21), ci=boot_gini(y21, p21, n=300)),
])
out["Gini"] = out["Gini"].round(3); out["ci"] = out["ci"].apply(lambda t: f"{t[0]:.3f}–{t[1]:.3f}")
out""")
md("**Калибровка по децилям** (когорта 2023, 12 месяцев). Средний PD против фактической доли дефолтов:")
code("""dt = decile_table(y12, p12); dt["разрыв, п.п."] = ((dt.mean_pd - dt.default_rate) * 100).round(2); dt.round(4)""")
md("**Уровни A–D на 2023** (доля дефолтов должна расти):")
code("""pd.concat([level_table(y12, model.level(p12)).assign(окно="12 мес."), level_table(y21, model.level(p21)).assign(окно="21 мес.")]).round(4)""")
md("**Стабильность (PSI):** скор между обучением (вне обучающих частей) и проверкой, и самые нестабильные признаки.")
code("""psi_s = psi(oof_pd, p12, edges=np.quantile(oof_pd, np.linspace(0, 1, 11)))
psi_f = pd.Series({c: psi(Xa[c].astype(float), Xb[c].astype(float)) for c in NUM_COLS}).sort_values(ascending=False)
print("PSI скора: %.4f (порог 0,25)" % psi_s); psi_f.head(5).round(4)""")
md("""## 9. Проверка на утечку
Все признаки считаются из отчётности за год T (с лагами T−1, T−2); данных после даты среза в признаках нет: поля витрины с датами ликвидации и событиями убраны из выборки на этапе `01_extract_cohorts.py`. Что проверено (результаты — в `out/pd_results.json`):
- возраст считается как год отчётности минус год регистрации; лаги T−1 и T−2 когорты 2023 совпадают со значениями когорты 2022 для 100% общих компаний;
- Gini на проверке 0,76: ниже порога подозрения 0,9; максимальный Gini одного признака 0,52;
- модель не держится на одном признаке: без маржи Gini 0,801, без маржи и абсолютной ликвидности 0,785, без возраста и признака «нет отчётности за прошлый год» 0,803;
- ранжирование сохраняется в течение окна: чем дальше от среза дефолт, тем ниже Gini (по кварталам 0,80 / 0,83 / 0,69 / 0,71).

Что проверить нельзя: дату подачи отчётности. Витрина её не хранит, поэтому не доказано, что отчётность года T была подана до даты среза. Поле витрины `pit_status` само помечено как непроверенное.""")
code("""aud = json.load(open(OUT + "pd_results.json"))["leakage_audit"]
{k: (round(v, 3) if isinstance(v, float) else v) for k, v in aud.items() if k.startswith("cv_gini") or k.startswith("max_")}""")
md("## 10. SHAP: главные факторы и `score(inn)`")
code("""sample = np.random.default_rng(1).choice(len(Xb[mb]), 20000, replace=False)
sv = model.contributions(Xb[mb].iloc[sample])[:, :-1]
imp = pd.Series(np.abs(sv).mean(0), index=FEATURES).sort_values(ascending=False)
imp.index = [RU_NAMES.get(i, i) for i in imp.index]; imp.head(10).round(3)""")
code("""import os
os.environ["PD_STORE_PATHS"] = os.path.join(ROOT, "data", "scoring_store_2024.parquet")
os.environ["PD_MODEL_PATH"] = os.path.join(ROOT, "out", "pd_model.pkl")
from score_inn import score, score_many
store_inns = pd.read_parquet(os.environ["PD_STORE_PATHS"], columns=["inn", "line_2110"])
demo = store_inns[store_inns.line_2110 >= 800000].inn.sample(3, random_state=5).tolist()
for i in demo:
    r = score(i); print(i, "PD=%.2f%%" % (r["pd"] * 100), "уровень", r["level"]); [print("   ", f["text"]) for f in r["factors"]]
print(score("0000000000"))""")
md("## 11. Графики")
code("""from IPython.display import Image
Image(OUT + "pd_report_plots.png")""")
md("## 12. Варианты метки Б, А, 0 на кросс-валидации внутри 2022\nОдна и та же модель и признаки, одни и те же части; таблица из скрипта `06_label_variants.py`:")
code("""pd.read_csv(OUT + "label_variants_cv.csv").round(4)""")
md("""Вариант Б (банкротство и исключение) лучше всего предсказывает «любой дефолт» (Gini 0,81) и теряет около 0,03 на чистых банкротствах по сравнению с моделью, обученной только на банкротствах (0,75 против 0,78). Модель, обученная только на банкротствах, на исключениях заметно слабее (0,68). Вариант 0 (любая ликвидация, включая реорганизации и добровольные) слабее по обеим целям, особенно на банкротствах (0,43). Выбор варианта Б по ТЗ подтверждается.""")
nb["cells"] = C
nbf.write(nb, "pd_model_training.ipynb")
