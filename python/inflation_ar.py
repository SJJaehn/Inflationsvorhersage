"""
inflation_ar.py  (port of inflation_prediction_ar.m / bKap4)

Pure autoregressive one-step-ahead OOS inflation forecast. The target is
predicted from its own past values only, with a reporting lag r that accounts
for publication delay: standing at date t the most recent usable value is
y[t-(r+1)].

  Regressors for y[t]:  y[t-(r+1)], ..., y[t-(r+p)]
      r = REPORT_LAG, p = lookback (number of AR lags)

The lookback p can be fixed or selected each window by AIC (statsmodels OLS).
Benchmarked against the historical-mean forecast.

CSV format: col 0 = date, col 1 = target.
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
COUNTRY    = util.cfg("COUNTRY", "UK-Adj")            # "US" or "UK"
CSV_PATH   = f"./DATA/Liedtke/{COUNTRY}/aggregated.csv"
OUTPUT_DIR = "./RESULTS/"

ROLLING          = util.cfg("ROLLING", True)
TRAIN_OBS        = int(util.cfg("TRAIN_OBS", 120))
REPORT_LAG       = 1
TIME_LAG         = 1
LOOKBACK         = int(util.cfg("LOOKBACK", 12))   # p: AR lags, used when OPTIMAL_LOOKBACK is False
OPTIMAL_LOOKBACK = False       # select p by AIC each window
LOOKBACK_GRID    = range(1, 13)
PLOT_LAG_GRID    = True        # produce a chart comparing R² and RMSE% across all lags
# =========================================================================


def lag_matrix(y, lags):
    """Column k holds y shifted down by lags[k] (NaN-padded at the top)."""
    return np.column_stack([util.apply_lag(y, L) for L in lags])


def fit_predict(yin, Xin, xout):
    """OLS (const + Xin) on the window, return the one-step forecast at xout."""
    res = sm.OLS(yin, sm.add_constant(Xin)).fit()
    return res, float(res.predict(np.r_[1.0, xout])[0])


def _run_ar_for_lag(p, y, reg, train_obs, rolling):
    """Run the AR OOS loop for a fixed lookback p. Returns (yhat, yhat_bm)."""
    n = len(y)
    yhat = np.full(n, np.nan)
    yhat_bm = np.full(n, np.nan)
    for t in range(train_obs - 1, n - 1):
        idx = util.window_index(t, train_obs, rolling)
        i_out = t + 1
        yin = y[idx]
        yhat_bm[i_out] = yin.mean()
        Xin = reg[idx, :p]
        xout = reg[i_out, :p]
        res = sm.OLS(yin, sm.add_constant(Xin)).fit()
        yhat[i_out] = float(res.predict(np.r_[1.0, xout])[0])
    return yhat, yhat_bm


def plot_lag_grid(y, reg, out_dir):
    """Run the AR model for each p in LOOKBACK_GRID, collect R² and RMSE%,
    and save a two-panel chart (R² by lag, RMSE% by lag)."""
    lags, r2s, rmses = [], [], []
    for p in LOOKBACK_GRID:
        print(f"  Lag grid: p={p}...", end=" ", flush=True)
        yhat, yhat_bm = _run_ar_for_lag(p, y, reg, TRAIN_OBS, ROLLING)
        oos = util.evaluate_oos(y, yhat_bm, yhat)
        fq = util.forecast_quality(y, yhat)
        lags.append(p)
        r2s.append(oos["R2OOS"] * 100)
        rmses.append(fq["RMSE"] * 100)
        print(f"R²={oos['R2OOS']*100:.2f}%  RMSE={fq['RMSE']*100:.3f}%")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    colors = ["#b30000" if v > 0 else "#3a6ea5" for v in r2s]
    ax1.bar(lags, r2s, color=colors)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_ylabel("OOS R² (in %)", fontsize=11)
    ax1.set_title(f"AR-Modell nach Lag  —  {COUNTRY}", fontsize=12)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.bar(lags, rmses, color="#3a6ea5")
    ax2.set_ylabel("RMSE (in %)", fontsize=11)
    ax2.set_xlabel("AR-Lags (p)", fontsize=11)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_xticks(lags)

    fig.tight_layout()
    path = os.path.join(out_dir, "lag_grid_chart.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Lag-grid chart saved to: {path}")


def main():
    dates, y, _, _ = util.load_data(CSV_PATH)
    print(f"Reporting lag r = {REPORT_LAG} -> first usable lag is y[t-{REPORT_LAG + TIME_LAG}]")

    p_max = max(LOOKBACK_GRID) if (OPTIMAL_LOOKBACK or PLOT_LAG_GRID) else LOOKBACK
    reg = lag_matrix(y, [REPORT_LAG + k for k in range(TIME_LAG, TIME_LAG + p_max)])

    # Pre-filter to rows where y and all lags are complete.
    # Expanding windows work correctly because the filtered series has no gaps.
    ok = ~np.isnan(y) & ~np.isnan(reg).any(axis=1)
    dates = dates[ok]
    y     = y[ok]
    reg   = reg[ok]
    n = len(y)
    print(f"Loaded {n} complete observations (filtered from {ok.size})")

    yhat = np.full(n, np.nan)
    yhat_bm = np.full(n, np.nan)
    lag_used = np.full(n, np.nan)

    for t in range(TRAIN_OBS - 1, n - 1):
        idx = util.window_index(t, TRAIN_OBS, ROLLING)
        i_out = t + 1
        yin = y[idx]
        yhat_bm[i_out] = yin.mean()

        if OPTIMAL_LOOKBACK:
            best_aic, best_p, best_res, best_xout = np.inf, None, None, None
            for p in LOOKBACK_GRID:
                Xin = reg[idx, :p]
                xout = reg[i_out, :p]
                res = sm.OLS(yin, sm.add_constant(Xin)).fit()
                if res.aic < best_aic:
                    best_aic, best_p, best_res, best_xout = res.aic, p, res, xout
            if best_res is None:
                continue
            p_use, res, xout = best_p, best_res, best_xout
        else:
            p_use = LOOKBACK
            Xin = reg[idx, :p_use]
            xout = reg[i_out, :p_use]
            res = sm.OLS(yin, sm.add_constant(Xin)).fit()

        yhat[i_out] = float(res.predict(np.r_[1.0, xout])[0])
        lag_used[i_out] = p_use

    oos = util.evaluate_oos(y, yhat_bm, yhat)
    fq = util.forecast_quality(y, yhat)
    valid = ~(np.isnan(y) | np.isnan(yhat))
    beg, end = util.oos_date_range(dates, valid)

    print("\n========== AR model metrics ==========")
    print(f"  OOS obs    : {int(valid.sum())}")
    if OPTIMAL_LOOKBACK:
        print(f"  Lookback   : {np.nanmin(lag_used):.0f} to {np.nanmax(lag_used):.0f} "
              f"(median {np.nanmedian(lag_used):.0f})")
    print(f"  OOS R2     : {oos['R2OOS']:.6f}")
    print(f"  CW stat/p  : {oos['CW']:.4f} / {oos['CWp']:.4f}")
    print(f"  DM stat/p  : {oos['DM']:.4f} / {oos['DMp']:.4f}")
    print(f"  RMSE / MAE : {fq['RMSE']:.6f} / {fq['MAE']:.6f}")
    print(f"  Cor/Hit    : {fq['Cor']:.6f} / {fq['HitRate']:.6f}")
    print(f"  OOS period : {beg} to {end}")

    lookback_str = f"optimal[{min(LOOKBACK_GRID)}-{max(LOOKBACK_GRID)}]" \
        if OPTIMAL_LOOKBACK else str(LOOKBACK)
    lookback_tag = f"optimal{min(LOOKBACK_GRID)}-{max(LOOKBACK_GRID)}" \
        if OPTIMAL_LOOKBACK else f"p{LOOKBACK}"
    options = (f"train{TRAIN_OBS}_{util.window_tag(ROLLING)}"
               f"_report{REPORT_LAG}_{lookback_tag}")
    out_dir = util.result_dir(OUTPUT_DIR, "AR", COUNTRY, "oos", options)

    summary = {
        "WindowType": "rolling" if ROLLING else "expanding",
        "TrainObs": TRAIN_OBS, "ReportLag": REPORT_LAG, "Lookback": lookback_str,
        "OOS_Beg": beg, "OOS_End": end, "Num_OOS_Obs": int(valid.sum()),
        "R2_OOS": oos["R2OOS"], "CW_stat": oos["CW"], "CW_p": oos["CWp"],
        "DM_stat": oos["DM"], "DM_p": oos["DMp"],
        "RMSE": fq["RMSE"], "MAE": fq["MAE"], "Cor": fq["Cor"], "HitRate": fq["HitRate"],
        "R2_MZ": fq["MZ_R2"], "F_MZ": fq["MZ_F"], "p_MZ": fq["MZ_p"],
    }
    sum_path, _ = util.save_summary(out_dir, "ar", summary)

    pred_path = os.path.join(out_dir, "predictions.csv")
    pd.DataFrame({"Date": dates, "Actual": y, "Forecast": yhat,
                  "Benchmark": yhat_bm, "LookbackUsed": lag_used}
                 ).to_csv(pred_path, index=False)

    print(f"\nSummary saved to:     {sum_path}")
    print(f"Predictions saved to: {pred_path}")

    if PLOT_LAG_GRID:
        grid_dir = util.result_dir(
            OUTPUT_DIR, "AR", COUNTRY, "oos",
            f"train{TRAIN_OBS}_{util.window_tag(ROLLING)}_report{REPORT_LAG}_grid")
        print("\n--- Lag-grid comparison ---")
        plot_lag_grid(y, reg, grid_dir)


if __name__ == "__main__":
    main()
