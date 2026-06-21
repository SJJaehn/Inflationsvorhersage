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

import util

# =========================================================================
#  CONFIG
# =========================================================================
CSV_PATH   = "./DATA/Liedtke/US/aggregated.csv"
OUTPUT_DIR = "./RESULTS/"

MODE             = "oos"   # "oos" or "insample"

ROLLING          = True    # (oos) rolling vs expanding window
TRAIN_OBS        = 60      # (oos) in-sample window length
TIME_LAG         = 1       # predictor lag (1 = standard predictive regression)
SHARED_TIMEFRAME = False    # evaluate every predictor on the SAME complete sample
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


def main():
    if MODE == "oos":
        out, prefix = run_oos()
    elif MODE == "insample":
        out, prefix = run_insample()
    else:
        raise ValueError(f"Unknown MODE: {MODE}")

    util.ensure_dir(OUTPUT_DIR)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, f"{prefix}_results_{ts}.csv")
    out.to_csv(path, index=False)
    print(f"\nResults saved to: {path}")


if __name__ == "__main__":
    main()
