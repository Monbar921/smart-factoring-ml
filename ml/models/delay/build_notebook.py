import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# Модель задержки оплаты по УПД: обучение и проверка
Версия delay-1.0 · «Смарт-факторинг» · по ТЗ «Обучение модели задержки оплаты по УПД» v1.2, раздел 8

Ноутбук показывает выборку, признаки, параметры, результаты на проверочном периоде, ROC, калибровку по децилям, важность признаков, смещение `expected_days` и проверки на утечку. Долгие шаги (подбор параметров на всех десяти партиях, обучение, проверка) выполнены скриптами `dl_dataset.py`, `dl_train.py`, `dl_eval.py`, их результаты лежат в `art/`; здесь они показаны, а выборка, признаки и `score_delay` пересчитываются вживую на тестовой партии.

**Что значат метрики.** Реальных оплат УПД у нас нет, данные синтетические. Метрики показывают, что модель восстанавливает заложенные в данные зависимости и не смотрит в будущее. О качестве на реальных оплатах они ничего не говорят.

**Правило.** Проверочный период (месяцы 25–29) в подборе параметров, калибровке и выборе признаков не участвовал.""")
code("""import json, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from IPython.display import Image, display
import dl_common as dc, dl_dataset as dd, dl_model as dm, score_delay as sd
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
R = json.load(open("art/eval_results.json", encoding="utf-8"))
S = json.load(open("art/search.json", encoding="utf-8"))
DS = json.load(open("art/dataset_summary.json", encoding="utf-8"))
TEST_BATCH = os.environ.get("DELAY_TEST_DIR", "../delivery/batch_01")
TRUTH = os.environ.get("DELAY_TRUTH", "../full/truth_upd.parquet")
print("тестовая партия:", TEST_BATCH)""")

md("""## 1. Данные и сверка оплат
Партия читается из CSV по контракту данных, как её увидит продукт. Оплаты распределяются по УПД правилами Б.2 (номер УПД, номер договора, самый старый открытый, остаток переносится дальше). Из этого получаем дату закрытия каждого УПД.""")
code("""tab = dc.load_csv_batch(TEST_BATCH)
U, A = dc.reconcile(tab)
print({k: len(v) for k, v in tab.items()}, "| событий распределения:", len(A))
tr = pd.read_parquet(TRUTH).set_index("upd_id")
m = U.set_index("upd_id").join(tr[["full_pay_date", "open_at_d_end"]])
closed = m.close_d.notna()
same = (m.close_d[closed].values == dc.day_int(m.full_pay_date[closed])).mean()
print("УПД: %d, закрыто на D_end: %d, открыто: %d" % (len(U), closed.sum(), (~closed).sum()))
print("даты закрытия по сверке совпадают с правдой генератора: %.1f%%" % (100 * same))""")

md("""## 2. Выборка: строки, метки, части по времени
Строка — УПД и дата `t`: дата подписания и далее каждые 14 дней, пока УПД открыт и срок не наступил. Все строки одного УПД попадают в одну часть (по дате подписания). Метка «задержка больше k дней»: 1, если УПД закрыт с такой задержкой или остаётся открытым дольше k дней после плановой даты; 0, если закрыт с задержкой не больше k; иначе строку для этого k не берём. Метки считаются по наблюдаемым данным, правда не используется.""")
code("""b = dd.part_bounds()
part = dd.part_of(U.upd_d.values, b)
rows = dc.make_rows(U, b["d_end"], mask=(part != "history"))
lab = dc.make_labels(U, rows, b["d_end"])
rows["part"] = part[rows.u.values]
print("тестовая партия, строк по частям:", rows.part.value_counts().to_dict())
pd.concat([rows.head(6), lab.head(6)], axis=1)""")
code("""t = pd.DataFrame(DS["parts"]).T.rename(columns={"rows": "строк", "upds": "УПД", "rows_no_history": "строк no_history", "rows_sign": "строк подписания"})
t.index = [{"train": "обучение (месяцы 4–20)", "gap": "пропуск (21–24)", "test": "проверка (25–29)", "rest": "остаток окна (30–33)"}[i] for i in t.index]
print("границы частей:", DS["bounds_dates"])
t""")
code("""pd.DataFrame({"обучение": {k: v["rate"] for k, v in DS["labels_train"].items()}, "проверка": {k: v["rate"] for k, v in DS["labels_test"].items()}}).rename_axis("задержка больше, дн.").round(4)""")

md("""## 3. Признаки
Только то, что известно на дату `t`: УПД и платежи не позже `t`. Признаки истории считаются по закрытым УПД пары и покупателя **внутри партии**. Не берём: тип стоп-фактора и сам стоп-фактор, содержимое `truth_upd`, идентификаторы.""")
code("""sub = rows[rows.part != "history"].sample(2000, random_state=1)
X = dc.features(tab, U, A, sub, sorted(tab["cl"].industry.unique()))
print(len(dc.FEATURES), "признаков")
X[dc.FEATURES].describe().T[["count", "mean", "min", "50%", "max"]].round(3)""")

md("## 4. Подбор параметров и калибровка\nКросс-валидация на обучающем периоде: пять частей, группы по УПД. Три конфигурации бустинга на каждый порог, число деревьев по ранней остановке.")
code("""rows_ = []
for k, v in S["search"].items():
    rows_.append(dict(порог=int(k), конфигурация=v["config"], деревьев=v["iters"], logloss=v["cv"]["logloss"], AUC_cv=v["cv"]["auc_sign"], AUC_наивного=v["cv"]["auc_naive_sign"], AUC_потолка=v["cv"]["auc_ceiling_sign"]))
pd.DataFrame(rows_).round(4)""")
code("""pd.DataFrame({k: {**{a: round(b, 5) for a, b in v["logloss"].items()}, "выбрано": v["kind"]} for k, v in S["calibration"].items()}).T.rename_axis("порог")""")

md("## 5. Проверка: месяцы 25–29\nМетрики на строках, где `t` — дата подписания УПД (одна строка на УПД). Потолок — AUC скрытых вероятностей генератора на тех же строках.")
code("""pd.DataFrame({k: {"AUC модели": v["model"], "наивное правило": v["naive"], "потолок": v["ceiling"], "прирост к наивному": v["model"] - v["naive"], "УПД": v["n_sign"], "доля задержек": v["pos_rate"]} for k, v in R["auc"].items()}).T.rename_axis("порог").round(4)""")
code("""display(Image("art/figs/roc.png"))""")
md("**Калибровка по децилям** (средняя оценка против фактической доли задержек в дециле):")
code("""display(Image("art/figs/calibration.png"))
pd.DataFrame({k: {"ECE (строки подписания)": v["ece_sign"], "ECE (все строки)": v["ece_all"], "средняя оценка": v["mean_pred_sign"], "доля задержек": v["mean_obs_sign"]} for k, v in R["calibration"].items()}).T.round(4)""")
md("**Монотонность.** Средняя `p_14` по уровням покупателя A–D и по пяти группам суммы УПД к обычной для пары:")
code("""display(pd.DataFrame(R["monotony"]["level"]).rename(columns={"index": "уровень", "p14": "средняя p_14", "obs": "факт"}).round(4))
pd.DataFrame(R["monotony"]["amount"]).rename(columns={"aq": "группа", "p14": "средняя p_14", "obs": "факт", "ratio": "сумма к обычной"}).round(4)""")
md("**Важность признаков** (доля прироста качества по деревьям):")
code("""display(Image("art/figs/importance.png"))""")

md("""## 6. Ожидаемые дни
Регрессия на закрытых УПД. Смещение из-за того, что берутся только закрытые, оценено по `truth_upd`: в проверочном периоде ещё открытые УПД имеют самые длинные задержки.""")
code("""e = R["expected_days"]
pd.Series({
 "смещение к закрытым УПД, дн.": e["sign_bias_model"],
 "смещение к полной правде, дн.": e["sign_bias_vs_truth_all"],
 "средняя задержка проверки: закрытые / вся правда": "%.2f / %.2f" % (e["selection_test"]["mean_true_closed"], e["selection_test"]["mean_true_all"]),
 "средняя задержка обучения: закрытые / вся правда": "%.2f / %.2f" % (e["selection_train"]["mean_true_closed"], e["selection_train"]["mean_true_all"]),
 "MAE модели / медиана по паре / ноль": "%.2f / %.2f / %.2f" % (e["sign_mae_model"], e["sign_mae_pair_median"], e["sign_mae_zero"]),
 "RMSE модели / среднее по паре / медиана по паре": "%.2f / %.2f / %.2f" % (e["sign_rmse_model"], e["sign_rmse_pair_mean"], e["sign_rmse_pair_median"]),
 "RMSE точного среднего генератора": e["sign_rmse_oracle_mean"],
 "корреляция с точным средним генератора": e["sign_corr_model_oracle"]}).to_frame("значение")""")
code("""display(Image("art/figs/expected_days.png"))""")

md("## 7. Проверка на утечку, скорость, повторяемость")
code("""print("перемешанные метки обучения, p_14, пять запусков:", [round(x, 3) for x in R["shuffle"]["train_shuffled_each"]], "среднее", round(R["shuffle"]["train_shuffled"], 3))
print("перемешанные метки проверки, p_14:", round(R["shuffle"]["test_labels_shuffled"], 3))
print("признаки по урезанным до t таблицам:", R["truncation_test"])
print("скорость:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in R["speed"].items()})""")
md("**Те же оценки вживую.** `score_delay` по CSV тестовой партии на `D_end` и сверка с файлом `delay_scores.csv` партии:")
code("""model = sd.get_model()
ids = pd.read_csv(os.path.join(TEST_BATCH, "delay_scores.csv"), dtype=str).upd_id.tolist()
import time; t0 = time.time()
out = sd.score_delay(ids, "2025-12-30", TEST_BATCH, model)
ref = pd.read_csv(os.path.join(TEST_BATCH, "delay_scores.csv"), dtype={"upd_id": str})
cols = ["p_7", "p_14", "p_30", "p_90", "expected_days"]
diff = (out[cols].astype(float).round({"p_7": 6, "p_14": 6, "p_30": 6, "p_90": 6, "expected_days": 2}).fillna(-1).values - ref[cols].astype(float).fillna(-1).values)
print("УПД: %d за %.2f с; макс. расхождение с delay_scores.csv: %.6f; статусы: %s" % (len(out), time.time() - t0, np.abs(diff).max(), out.status.value_counts().to_dict()))
out.head()""")

md("## 8. Итоги по критериям раздела 8.5")
code("""cr = pd.DataFrame(R["criteria"]); cr["итог"] = np.where(cr.passed, "прошла", "НЕ ПРОШЛА")
cr[["name", "requirement", "value", "итог"]].rename(columns={"name": "проверка", "requirement": "требование", "value": "значение"})""")
md("""Не выполнены три пункта: прирост к наивному правилу по `p_14` (+0,014 вместо +0,03), MAE `expected_days` против медианы по паре (среднее по определению проигрывает медиане по MAE, у точного среднего генератора тоже 10,09), К-11 в трёх партиях (отдельный отчёт о покрытии). Подробности и причины: `Delay_Model_Report.md`.

**Ограничения.** Данные синтетические. Оценки в проверочные месяцы в среднем на 1–2 п.п. ниже фактической доли: задержек стало больше, чем в обучении. Для УПД, у которых нет закрытых УПД ни у пары, ни у покупателя, функция возвращает статус `no_history` без вероятностей. Пара без истории, но с покупателем с историей, оценивается по истории покупателя.""")
nb["cells"] = C
nbf.write(nb, "delay_model_training.ipynb")
print("ok", len(C))
