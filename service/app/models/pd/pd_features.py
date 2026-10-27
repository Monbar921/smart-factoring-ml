"""
Feature builder for the buyer PD model (TZ PD v1.1, section 6).

Input  : company-year table with RFSD lines for year T and lags _t1 (T-1), _t2 (T-2),
         plus line_2120 for year T, okved_section, region, simplified, age.
         Amounts are in thousand RUB.
Output : model features. Only information known at the cutoff date (reports up to year T).
Rules  : division by zero -> NaN; ratios clipped at the 1st/99th percentile fitted on the
         training cohort (2022); no ЕФРСБ / court / tax features.
"""
import numpy as np
import pandas as pd

LINES = ["2110", "1200", "1250", "1500", "1300", "1600", "1410", "1510", "2400", "2200", "2330", "1230", "1520"]

# ratios that are clipped (name -> kept as-is otherwise)
CLIP_COLS = [
    "net_margin", "debt_to_revenue", "receivables_days", "payables_days", "equity_change_1y",
    "interest_coverage", "revenue_growth_1y", "cash_ratio", "current_ratio", "equity_ratio",
    "current_ratio_change_1y", "revenue_growth_2y", "equity_change_2y", "current_ratio_change_2y",
]
NUM_COLS = [
    "log_revenue", "log_assets", "age",
    "net_margin", "debt_to_revenue", "receivables_days", "payables_days", "equity_change_1y",
    "interest_coverage", "revenue_growth_1y", "cash_ratio", "current_ratio", "equity_ratio",
    "current_ratio_change_1y", "revenue_growth_2y", "equity_change_2y", "current_ratio_change_2y",
]
FLAG_COLS = [
    "simplified_form", "no_prev_report", "receivables_faster_than_revenue", "payables_faster_than_revenue",
    "loss", "two_years_losses", "negative_equity", "revenue_drop_30", "is_moscow_mo",
]
CAT_COLS = ["okved_section", "region_group"]
FEATURES = NUM_COLS + FLAG_COLS + CAT_COLS

# plain-language names for explanations (used in SHAP top-3 and LLM texts)
RU_NAMES = {
    "log_revenue": "размер выручки", "log_assets": "размер активов", "age": "возраст компании",
    "net_margin": "маржа чистой прибыли", "debt_to_revenue": "займы к выручке",
    "receivables_days": "дни оборота дебиторской задолженности", "payables_days": "дни оборота кредиторской задолженности",
    "equity_change_1y": "изменение капитала за год", "interest_coverage": "покрытие процентов",
    "revenue_growth_1y": "изменение выручки за год", "cash_ratio": "абсолютная ликвидность",
    "current_ratio": "текущая ликвидность", "equity_ratio": "доля собственного капитала",
    "current_ratio_change_1y": "изменение текущей ликвидности за год", "revenue_growth_2y": "изменение выручки за два года",
    "equity_change_2y": "изменение капитала за два года", "current_ratio_change_2y": "изменение текущей ликвидности за два года",
    "simplified_form": "упрощённая отчётность", "no_prev_report": "нет отчётности за прошлый год",
    "receivables_faster_than_revenue": "дебиторка растёт быстрее выручки",
    "payables_faster_than_revenue": "кредиторка растёт быстрее выручки", "loss": "убыток за год",
    "two_years_losses": "убыток два года подряд", "negative_equity": "отрицательный капитал",
    "revenue_drop_30": "выручка упала более чем на 30%", "is_moscow_mo": "регион: Москва или область",
    "okved_section": "вид деятельности (раздел ОКВЭД)", "region_group": "регион",
}


def _div(a, b):
    """a / b with division by zero -> NaN (inf -> NaN)."""
    b = b.where(b != 0)
    r = a / b
    return r.replace([np.inf, -np.inf], np.nan)


def _log(x):
    return np.log(x.where(x > 0))


def _region_group(region):
    r = region.fillna("")
    out = pd.Series("other", index=region.index)
    out[r == "moscow city"] = "moscow_city"
    out[r == "moscow reg."] = "moscow_region"
    out[r == "sankt-petersburg"] = "spb"
    return out


def raw_ratios(d: pd.DataFrame) -> pd.DataFrame:
    """Unclipped features from raw RFSD lines."""
    g = lambda c: d[c].astype(float)
    f = pd.DataFrame(index=d.index)
    rev, rev1, rev2 = g("line_2110"), g("line_2110_t1"), g("line_2110_t2")
    ta = g("line_1600")
    f["log_revenue"] = _log(rev)
    f["log_assets"] = _log(ta)
    f["age"] = g("age")
    f["net_margin"] = _div(g("line_2400"), rev)
    f["debt_to_revenue"] = _div(g("line_1410") + g("line_1510"), rev)
    f["receivables_days"] = _div(g("line_1230"), rev) * 365
    f["payables_days"] = _div(g("line_1520"), g("line_2120").abs()) * 365
    f["equity_change_1y"] = _div(g("line_1300") - g("line_1300_t1"), ta)
    f["interest_coverage"] = _div(g("line_2200"), g("line_2330").abs())
    f["revenue_growth_1y"] = _div(rev, rev1) - 1
    f["cash_ratio"] = _div(g("line_1250"), g("line_1500"))
    cr = _div(g("line_1200"), g("line_1500"))
    cr1 = _div(g("line_1200_t1"), g("line_1500_t1"))
    cr2 = _div(g("line_1200_t2"), g("line_1500_t2"))
    f["current_ratio"] = cr
    f["equity_ratio"] = _div(g("line_1300"), ta)
    f["current_ratio_change_1y"] = cr - cr1
    f["revenue_growth_2y"] = _div(rev, rev2) - 1
    f["equity_change_2y"] = _div(g("line_1300") - g("line_1300_t2"), ta)
    f["current_ratio_change_2y"] = cr - cr2

    # flags (0/1; NaN-safe: unknown -> 0 for growth-comparison flags)
    rec_g = _div(g("line_1230"), g("line_1230_t1")) - 1
    pay_g = _div(g("line_1520"), g("line_1520_t1")) - 1
    rev_g = f["revenue_growth_1y"]
    f["simplified_form"] = g("simplified").astype(int)
    f["no_prev_report"] = (rev1.isna() & g("line_1600_t1").isna()).astype(int)
    f["receivables_faster_than_revenue"] = (rec_g > rev_g).fillna(False).astype(int)
    f["payables_faster_than_revenue"] = (pay_g > rev_g).fillna(False).astype(int)
    loss = (g("line_2400") < 0)
    loss1 = (g("line_2400_t1") < 0)
    f["loss"] = loss.astype(int)
    f["two_years_losses"] = (loss & loss1).astype(int)
    f["negative_equity"] = (g("line_1300") < 0).astype(int)
    f["revenue_drop_30"] = (rev_g < -0.30).fillna(False).astype(int)
    f["is_moscow_mo"] = d["region"].isin(["moscow city", "moscow reg."]).astype(int)
    f["okved_section"] = d["okved_section"].fillna("NA").astype(str)
    f["region_group"] = _region_group(d["region"])
    return f


class FeatureBuilder:
    def __init__(self, q_low=0.01, q_high=0.99):
        self.q_low, self.q_high = q_low, q_high
        self.bounds = {}
        self.okved_levels = None
        self.region_levels = ["moscow_city", "moscow_region", "spb", "other"]

    def fit(self, d: pd.DataFrame):
        f = raw_ratios(d)
        for c in CLIP_COLS:
            self.bounds[c] = (float(f[c].quantile(self.q_low)), float(f[c].quantile(self.q_high)))
        self.okved_levels = sorted(f["okved_section"].unique())
        return self

    def transform(self, d: pd.DataFrame, as_category=True) -> pd.DataFrame:
        f = raw_ratios(d)
        for c, (lo, hi) in self.bounds.items():
            f[c] = f[c].clip(lo, hi)
        if as_category:
            f["okved_section"] = pd.Categorical(f["okved_section"], categories=self.okved_levels)
            f["region_group"] = pd.Categorical(f["region_group"], categories=self.region_levels)
        return f[FEATURES]
