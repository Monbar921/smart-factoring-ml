import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pd_data import OUT
from pd_features import RU_NAMES
t1 = pd.read_csv(OUT + "levels_train_oof.csv"); t2 = pd.read_csv(OUT + "levels_2023_12m.csv")
s = pd.read_csv(OUT + "shap_mean_abs.csv", index_col=0).iloc[:, 0].head(12)[::-1]

def calib(ax):
    for tag, c, lab in (("2023_12m", "#1f77b4", "2023, окно 12 мес."), ("2023_21m", "#d62728", "2023, окно 21 мес.")):
        d = pd.read_csv(OUT + f"calibration_deciles_{tag}.csv")
        ax.plot(d.mean_pd * 100, d.default_rate * 100, "o-", color=c, label=lab)
    lo, hi = 0.03, 30
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="идеальная калибровка")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Средний PD в дециле, % (лог. шкала)"); ax.set_ylabel("Фактическая доля дефолтов, % (лог. шкала)")
    ax.set_title("Калибровка по децилям, когорта 2023"); ax.legend(); ax.grid(alpha=.3, which="both")

def levels(ax):
    w = .38; xs = np.arange(4)
    b1 = ax.bar(xs - w/2, t1.default_rate * 100, w, label="2022 (вне обучения)", color="#9ecae1")
    b2 = ax.bar(xs + w/2, t2.default_rate * 100, w, label="2023 (проверка)", color="#1f77b4")
    for r in list(b1) + list(b2): ax.text(r.get_x() + r.get_width()/2, r.get_height() + .2, f"{r.get_height():.1f}", ha="center", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels([f"{l}\n{sh*100:.0f}% компаний" for l, sh in zip(t2.level, t2.share)], fontsize=9)
    ax.set_ylabel("Доля дефолтов, %"); ax.set_title("Уровни риска A–D (окно 12 мес.)"); ax.legend(); ax.grid(alpha=.3, axis="y")

def shap(ax):
    ax.barh([RU_NAMES.get(i, i) for i in s.index], s.values, color="#2ca02c")
    ax.set_title("Топ признаков по SHAP"); ax.set_xlabel("Среднее |вклад| в лог-шансы"); ax.grid(alpha=.3, axis="x")

fig, ax = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True); calib(ax[0]); levels(ax[1]); shap(ax[2]); plt.savefig(OUT + "pd_report_plots.png", dpi=130); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True); calib(ax[0]); levels(ax[1]); plt.savefig(OUT + "pd_plot_calibration_levels.png", dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(8.5, 4.4), constrained_layout=True); shap(ax); plt.savefig(OUT + "pd_plot_shap.png", dpi=150); plt.close()
