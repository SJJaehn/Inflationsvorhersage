"""
inflation_single.py  (port of inflation_prediction_single.m / bKap3)

Predictive regression of inflation on EACH predictor on its own
(y[t] ~ const + x[t-lag]), in two modes:

  - MODE = "oos": rolling/expanding one-step-ahead OOS forecast, re-estimated
    each step (sklearn LinearRegression), benchmarked against the historical
    mean. Reports OOS R2, Clark-West, Diebold-Mariano, RMSE/MAE/Cor/HitRate
    and the Mincer-Zarnowitz regression.

  - MODE = "insample": one OLS fit per predictor on the whole sample
    (statsmodels OLS). Reports the slope coefficient + t-stat, R2 and the
    in-sample fit quality (RMSE/MAE/Cor/HitRate).

Results for all predictors are written to one CSV.

CSV format: col 0 = date, col 1 = target, col 2.. = predictors.
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

import util

# =========================================================================
#  CONFIG
# =========================================================================
COUNTRY    = util.cfg("COUNTRY", "US")            # "US" or "UK"
CSV_PATH   = f"./DATA/Liedtke/{COUNTRY}/aggregated.csv"
OUTPUT_DIR = "./RESULTS/"

MODE             = util.cfg("MODE", "oos")        # "oos" or "insample"

ROLLING          = util.cfg("ROLLING", True)      # (oos) rolling vs expanding window
TRAIN_OBS        = 120      # (oos) in-sample window length
TIME_LAG         = 1       # predictor lag (1 = standard predictive regression)
SHARED_TIMEFRAME = False    # evaluate every predictor on the SAME complete sample

PLOT_R2          = True     # bar chart of R2 per predictor (port of bKap3_4)
# =========================================================================


def prepare():
    dates, y, X_raw, pred_names = util.load_data(CSV_PATH)
    n_pred = X_raw.shape[1]
    print(f"Loaded {len(y)} observations, {n_pred} predictors from {CSV_PATH}")
    X_lag = util.apply_lag(X_raw, TIME_LAG)
    if SHARED_TIMEFRAME:
        ok = ~(np.isnan(y) | np.isnan(X_lag).any(axis=1))
        y, X_lag = y[ok], X_lag[ok, :]
        dates = dates[ok].reset_index(drop=True)
        print(f"Shared timeframe: {len(y)} of {len(ok)} observations retained.")
    return dates, y, X_lag, pred_names


def run_oos():
    dates, y, X_lag, pred_names = prepare()
    rows = []
    for j, name in enumerate(pred_names):
        yhat, yhat_bm = util.rolling_oos_forecast(
            y, X_lag[:, [j]], TRAIN_OBS, ROLLING)
        oos = util.evaluate_oos(y, yhat_bm, yhat)
        fq = util.forecast_quality(y, yhat)
        valid = ~(np.isnan(y) | np.isnan(yhat))
        beg, end = util.oos_date_range(dates, valid)
        rows.append({
            "Predictor": name, "OOS_Beg": beg, "OOS_End": end,
            "Num_OOS_Obs": int(valid.sum()),
            "R2_OOS": oos["R2OOS"], "CW_stat": oos["CW"], "CW_p": oos["CWp"],
            "DM_stat": oos["DM"], "DM_p": oos["DMp"],
            "RMSE": fq["RMSE"], "MAE": fq["MAE"], "Cor": fq["Cor"],
            "HitRate": fq["HitRate"],
            "R2_MZ": fq["MZ_R2"], "F_MZ": fq["MZ_F"], "p_MZ": fq["MZ_p"],
        })
        print(f"  [{j + 1}/{len(pred_names)}] {name:<28} "
              f"OOS R2={oos['R2OOS']:.4f} RMSE={fq['RMSE']:.4f} "
              f"MAE={fq['MAE']:.4f} Cor={fq['Cor']:.4f}")
    out = pd.DataFrame(rows)
    print("\n========== OOS summary ==========")
    print(out[["Predictor", "R2_OOS", "CW_p", "RMSE", "MAE", "Cor"]]
          .to_string(index=False))
    return out, "single_oos"


def run_insample():
    _, y, X_lag, pred_names = prepare()
    rows = []
    for j, name in enumerate(pred_names):
        x = X_lag[:, j]
        m = ~(np.isnan(y) | np.isnan(x))
        res = sm.OLS(y[m], sm.add_constant(x[m])).fit()
        yhat = np.asarray(res.fittedvalues)
        fq = util.forecast_quality(y[m], yhat)
        rows.append({
            "Predictor": name, "Num_Obs": int(m.sum()),
            "Intercept": res.params[0], "Beta": res.params[1],
            "t_Beta": res.tvalues[1], "p_Beta": res.pvalues[1],
            "R2": res.rsquared,
            "RMSE": fq["RMSE"], "MAE": fq["MAE"], "Cor": fq["Cor"],
            "HitRate": fq["HitRate"],
        })
        print(f"  [{j + 1}/{len(pred_names)}] {name:<28} "
              f"beta={res.params[1]:+.4g} (t={res.tvalues[1]:6.2f}) "
              f"R2={res.rsquared:.4f}")
    out = pd.DataFrame(rows)
    print("\n========== In-sample summary ==========")
    print(out[["Predictor", "Beta", "t_Beta", "R2", "RMSE", "Cor"]]
          .to_string(index=False))
    return out, "single_insample"


def plot_r2_bar(out, out_dir):
    """Bar chart of the R2 (in %) per predictor (port of the bKap3_4 diagram).

    OOS mode plots the OOS R2 vs the historical-mean benchmark; in-sample mode
    plots the regression R2. Bars are sorted descending so the strongest
    predictors are easy to read off. Saved as chart.png in `out_dir`.
    """
    col = "R2_OOS" if MODE == "oos" else "R2"
    ylabel = "OOS $R^2$ (in %)" if MODE == "oos" else "In-sample $R^2$ (in %)"

    d = out[["Predictor", col]].copy()
    d[col] = d[col] * 100
    d = d.sort_values(col, ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(d)), 5))
    colors = ["#2c7fb8" if v >= 0 else "#d95f5f" for v in d[col]]
    ax.bar(np.arange(len(d)), d[col], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(d)))
    ax.set_xticklabels(d["Predictor"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    path = os.path.join(out_dir, "chart.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"R2 bar chart saved to: {path}")


def main():
    if MODE == "oos":
        out, _ = run_oos()
        options = f"train{TRAIN_OBS}_{util.window_tag(ROLLING)}_lag{TIME_LAG}"
        mode_dir = "oos"
    elif MODE == "insample":
        out, _ = run_insample()
        options = f"lag{TIME_LAG}"
        mode_dir = "insample"
    else:
        raise ValueError(f"Unknown MODE: {MODE}")

    out_dir = util.result_dir(OUTPUT_DIR, "single", COUNTRY, mode_dir, options)
    path = os.path.join(out_dir, "results.csv")
    out.to_csv(path, index=False)
    print(f"\nResults saved to: {path}")

    if PLOT_R2:
        plot_r2_bar(out, out_dir)


if __name__ == "__main__":
    main()
