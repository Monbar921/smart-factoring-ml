"""Company-year table with model features and labels (Parquet) + field dictionary."""
import pandas as pd, numpy as np
from pd_data import *
from pd_features import FEATURES, RU_NAMES
a, b, fb, Xa, Xb = load()
parts = []
for coh, d, X, cutoff in ((2022, a, Xa, "2023-04-01"), (2023, b, Xb, "2024-04-01")):
    t = pd.DataFrame({"inn": d.inn.values, "cohort": coh, "financial_year": d.financial_year.values, "cutoff_date": cutoff,
                      "revenue_kRUB": d.line_2110.values})
    t = pd.concat([t, X.reset_index(drop=True)], axis=1)
    for c in ("y_B_12m", "y_A_12m", "y_0_12m", "y_B_21m", "y_A_21m", "y_0_21m"):
        t[c] = d[c].values if c in d.columns else np.nan
    for c in ("okved_section", "region_group"): t[c] = t[c].astype(str)
    parts.append(t)
T = pd.concat(parts, ignore_index=True)
for c in (2022, 2023):
    T[T.cohort == c].to_parquet(OUT + f"pd_company_year_features_{c}.parquet", index=False, compression="zstd", compression_level=12)
print(T.shape, T.groupby("cohort").size().to_dict())

F = {
 "log_revenue": "ln(выручка, стр. 2110)", "log_assets": "ln(активы, стр. 1600)", "age": "год отчётности − год регистрации",
 "net_margin": "2400 / 2110", "debt_to_revenue": "(1410 + 1510) / 2110", "receivables_days": "1230 / 2110 × 365",
 "payables_days": "1520 / |2120| × 365", "equity_change_1y": "(1300 за T − 1300 за T−1) / 1600 за T",
 "interest_coverage": "2200 / |2330|", "revenue_growth_1y": "2110 за T / 2110 за T−1 − 1", "cash_ratio": "1250 / 1500",
 "current_ratio": "1200 / 1500", "equity_ratio": "1300 / 1600", "current_ratio_change_1y": "текущая ликвидность T − T−1",
 "revenue_growth_2y": "2110 за T / 2110 за T−2 − 1", "equity_change_2y": "(1300 за T − 1300 за T−2) / 1600 за T",
 "current_ratio_change_2y": "текущая ликвидность T − T−2",
 "simplified_form": "1 — упрощённая отчётность", "no_prev_report": "1 — нет выручки и активов за T−1 (не сдавала или не существовала)",
 "receivables_faster_than_revenue": "1 — рост дебиторки (1230) за год больше роста выручки",
 "payables_faster_than_revenue": "1 — рост кредиторки (1520) за год больше роста выручки",
 "loss": "1 — чистый убыток (2400 < 0)", "two_years_losses": "1 — убыток за T и T−1", "negative_equity": "1 — капитал (1300) < 0",
 "revenue_drop_30": "1 — выручка упала больше чем на 30%", "is_moscow_mo": "1 — Москва или Московская область",
 "okved_section": "раздел ОКВЭД (буква)", "region_group": "moscow_city / moscow_region / spb / other",
}
rows = ["| Поле | Что это | Формула / значения |", "| --- | --- | --- |",
        "| `inn` | ИНН | |", "| `cohort` | Когорта (год отчётности) | 2022 — обучение, 2023 — проверка |",
        "| `financial_year` | Год отчётности T | |", "| `cutoff_date` | Дата среза | 01.04 года T+1 |",
        "| `revenue_kRUB` | Выручка, тыс. ₽ (стр. 2110), для сегмента от 800 млн ₽ | |"]
rows += [f"| `{k}` | {RU_NAMES[k]} | {F[k]} |" for k in FEATURES]
rows += ["| `y_B_12m`, `y_A_12m`, `y_0_12m` | Метки на окне 12 месяцев (пусто — строка не берётся) | см. ТЗ, раздел 4 |",
         "| `y_B_21m`, `y_A_21m`, `y_0_21m` | Метки на окне 21 месяц (только когорта 2023) | |"]
open(OUT + "pd_fields.md", "w").write("\n".join(rows) + "\n\nКоэффициенты обрезаны по 1-му и 99-му перцентилю обучающей когорты 2022; деление на ноль — пропуск. Суммы — тыс. ₽.\n")
print("ok")
