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
TRAIN_OBS        = int(util.cfg("TRAIN_OBS", 120))  # (oos) in-sample window length
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


# Orange->red colour ramp by significance, plus a colour for the
# non-significant bars. In-sample uses the |t| of the slope (two-sided normal
# critical values); OOS uses the Clark-West p-value (one-sided).
_SIG_COLORS = ["#fdae61", "#f16913", "#b30000"]   # 90% -> 95% -> 99%
_SIG_LABELS = ["90%", "95%", "99%"]
_SIG_NS = "#3a6ea5"                                # non-significant (steel blue)
_T_THRESH = [1.645, 1.960, 2.576]                  # |t| for 90/95/99% (two-sided)
_P_THRESH = [0.10, 0.05, 0.01]                     # p for 90/95/99%


def _sig_index_t(t):
    """Highest significance level cleared by |t| (-1 = none)."""
    if t is None or np.isnan(t):
        return -1
    idx = -1
    for i, thr in enumerate(_T_THRESH):
        if abs(t) >= thr:
            idx = i
    return idx


def _sig_index_p(p):
    """Highest significance level cleared by a p-value (-1 = none)."""
    if p is None or np.isnan(p):
        return -1
    idx = -1
    for i, thr in enumerate(_P_THRESH):
        if p <= thr:
            idx = i
    return idx


def plot_r2_bar(out, out_dir):
    """Bar chart of the R2 (in %) per predictor (port of the bKap3_4 diagram).

    OOS mode plots the OOS R2 vs the historical-mean benchmark; in-sample mode
    plots the regression R2. Bars are sorted descending and coloured on an
    orange->red scale by significance: in-sample by the slope's |t| (90/95/99%),
    OOS by the Clark-West p-value. Non-significant bars are steel blue. Saved as
    chart.png in `out_dir`.
    """
    col = "R2_OOS" if MODE == "oos" else "R2"
    title = f"{'Out-of-Sample' if MODE == 'oos' else 'In-Sample'} R² {COUNTRY}"

    # Pick the per-predictor significance measure available for this mode.
    if MODE == "oos":
        sig_col, sig_index, legend_title = "CW_p", _sig_index_p, "Signifikanz (CW)"
    else:
        sig_col, sig_index, legend_title = "t_Beta", _sig_index_t, "Signifikanz (|t|)"

    rmse_col = "RMSE"
    keep = (["Predictor", col, rmse_col]
            + ([sig_col] if sig_col in out.columns else []))
    d = out[[c for c in keep if c in out.columns]].copy()
    d[col] = d[col] * 100
    if rmse_col in d.columns:
        d[rmse_col] = d[rmse_col] * 100
    d = d.sort_values(col, ascending=False).reset_index(drop=True)

    have_sig = sig_col in d.columns
    have_rmse = rmse_col in d.columns
    if have_sig:
        colors = [_SIG_COLORS[i] if (i := sig_index(v)) >= 0 else _SIG_NS
                  for v in d[sig_col]]
    else:
        colors = ["#3a6ea5" if v >= 0 else "#d95f5f" for v in d[col]]

    fig, ax = plt.subplots(figsize=(max(10, 0.6 * len(d)), 5))
    ax.bar(np.arange(len(d)), d[col], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(d)))
    if have_rmse:
        tick_labels = [
            f"{util.short_name(p)}  |  R²: {r2:.1f}%  |  RMSE: {rmse:.2f}%"
            for p, r2, rmse in zip(d["Predictor"], d[col], d[rmse_col])
        ]
    else:
        tick_labels = [util.short_name(p) for p in d["Predictor"]]
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("R² (in %)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)

    if have_sig:
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=c, label=lbl)
                   for c, lbl in zip(_SIG_COLORS, _SIG_LABELS)]
        handles.append(Patch(facecolor=_SIG_NS, label="n.s."))
        ax.legend(handles=handles, title=legend_title,
                  fontsize=8, title_fontsize=8, frameon=False)

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
