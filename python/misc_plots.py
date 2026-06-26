"""
misc_plots.py
Generate comparison plots saved to RESULTS/misc/.

Reads from pre-computed result CSVs (AR and VAR). No new estimation is done.

Charts produced
---------------
r2_by_lag_UK_vs_US.png         UK vs US, AR + VARX, R² and RMSE
r2_by_lag_UK_vs_USAdj.png      UK vs US-Adj, AR + VARX, R² and RMSE
r2_by_lag_UKAdj_vs_USAdj.png   UK-Adj vs US-Adj, AR + VARX, R² and RMSE  [NEW]
r2_ar_timeframe.png             AR own-sample vs VARX-sample, UK and US
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "./RESULTS"
MISC_DIR    = "./RESULTS/misc"
LAGS        = list(range(1, 13))
os.makedirs(MISC_DIR, exist_ok=True)


# =========================================================================
#  Data loaders
# =========================================================================
def _read_kv(path):
    """Read a Key/Value CSV and return a dict {key: value_as_str}."""
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, index_col=0)
    return df["Value"].to_dict()


def load_ar(country, p):
    """AR result for country and lag count p (from inflation_ar.py output)."""
    path = os.path.join(RESULTS_DIR, "AR", country, "oos",
                        f"train120_rolling_report1_p{p}", "results.csv")
    return _read_kv(path)


def load_var(country, lags, model="VARX"):
    """AR or VARX component from inflation_var.py output, p=lags."""
    path = os.path.join(RESULTS_DIR, "VAR", country, "oos",
                        f"min120_rolling_lags{lags}_report1", "results.csv")
    d = _read_kv(path)
    prefix = model + "_"
    return {k[len(prefix):]: v for k, v in d.items() if k.startswith(prefix)}


def _safe(d, key):
    v = d.get(key)
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def collect(country, source):
    """
    source: "AR"     -> AR standalone results (inflation_ar.py)
            "AR_var" -> AR component inside VARX results (inflation_var.py)
            "VARX"   -> VARX component inside VARX results (inflation_var.py)
    Returns (r2_array, rmse_array) in %, length = len(LAGS).
    """
    r2s, rmses = [], []
    for p in LAGS:
        if source == "AR":
            d = load_ar(country, p)
            r2s.append(_safe(d, "R2_OOS") * 100)
            rmses.append(_safe(d, "RMSE") * 100)
        elif source == "AR_var":
            d = load_var(country, p, "AR")
            r2s.append(_safe(d, "R2_OOS") * 100)
            rmses.append(_safe(d, "RMSE") * 100)
        elif source == "VARX":
            d = load_var(country, p, "VARX")
            r2s.append(_safe(d, "R2_OOS") * 100)
            rmses.append(_safe(d, "RMSE") * 100)
    return np.array(r2s), np.array(rmses)


# =========================================================================
#  Chart helpers
# =========================================================================
_STYLES = {
    "solid":  dict(linestyle="-",  marker="o"),
    "dashed": dict(linestyle="--", marker="s"),
}
_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def _panel(ax, lags, series, ylabel, title, hline=True):
    """Draw one subplot panel with multiple series."""
    for i, (label, vals, style) in enumerate(series):
        kw = _STYLES[style]
        ax.plot(lags, vals, label=label, color=_COLORS[i], **kw)
    if hline:
        ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
    ax.set_xlabel("Number of lags", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(lags)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)


def comparison_chart(country1, country2, label1, label2, filename):
    """
    2-row × 2-col chart: rows = R² / RMSE, cols = AR / VARX.
    country1 solid blue, country2 dashed orange.
    """
    r2_ar1,  rmse_ar1  = collect(country1, "AR")
    r2_ar2,  rmse_ar2  = collect(country2, "AR")
    r2_vx1,  rmse_vx1  = collect(country1, "VARX")
    r2_vx2,  rmse_vx2  = collect(country2, "VARX")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(f"OOS R² and RMSE by Lag: {label1} vs {label2}", fontsize=13)

    _panel(axes[0, 0], LAGS,
           [(label1, r2_ar1, "solid"), (label2, r2_ar2, "dashed")],
           "OOS R² (%)", "AR model — R²")
    _panel(axes[0, 1], LAGS,
           [(label1, r2_vx1, "solid"), (label2, r2_vx2, "dashed")],
           "OOS R² (%)", "VARX model — R²")
    _panel(axes[1, 0], LAGS,
           [(label1, rmse_ar1, "solid"), (label2, rmse_ar2, "dashed")],
           "RMSE (%)", "AR model — RMSE", hline=False)
    _panel(axes[1, 1], LAGS,
           [(label1, rmse_vx1, "solid"), (label2, rmse_vx2, "dashed")],
           "RMSE (%)", "VARX model — RMSE", hline=False)

    fig.tight_layout()
    path = os.path.join(MISC_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def timeframe_chart():
    """
    AR R² and RMSE by lag: own-sample vs VARX-restricted-sample, for UK and US.
    2-row × 2-col: rows = R² / RMSE, cols = UK / US.
    """
    countries = [("UK", "UK"), ("US", "US")]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle("AR model — Own sample vs VARX sample", fontsize=13)

    for col, (country, label) in enumerate(countries):
        r2_own,  rmse_own  = collect(country, "AR")
        r2_var,  rmse_var  = collect(country, "AR_var")

        series_r2   = [(f"{label} — own sample",  r2_own,  "solid"),
                       (f"{label} — VARX sample", r2_var,  "dashed")]
        series_rmse = [(f"{label} — own sample",  rmse_own, "solid"),
                       (f"{label} — VARX sample", rmse_var, "dashed")]

        _panel(axes[0, col], LAGS, series_r2,   "OOS R² (%)",  f"{label} — R²")
        _panel(axes[1, col], LAGS, series_rmse, "RMSE (%)",    f"{label} — RMSE",
               hline=False)

    fig.tight_layout()
    path = os.path.join(MISC_DIR, "r2_ar_timeframe.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# =========================================================================
#  Main
# =========================================================================
if __name__ == "__main__":
    comparison_chart("UK",     "US",     "UK",     "US",     "r2_by_lag_UK_vs_US.png")
    comparison_chart("UK",     "US-Adj", "UK",     "US-Adj", "r2_by_lag_UK_vs_USAdj.png")
    comparison_chart("UK-Adj", "US-Adj", "UK-Adj", "US-Adj", "r2_by_lag_UKAdj_vs_USAdj.png")
    timeframe_chart()
    print("Done.")
