"""
inflation_full.py  (port of inflation_prediction_full.m / bKap3 kitchen sink)

"Kitchen-sink" inflation regression on ALL predictors at once, in two modes:

  - MODE = "oos": rolling/expanding one-step-ahead OOS forecast, re-estimated
    each step (sklearn LinearRegression), benchmarked against the historical
    mean. Reports OOS R2, Clark-West, Diebold-Mariano, RMSE/MAE/Cor/HitRate
    and the Mincer-Zarnowitz regression.

  - MODE = "insample": one OLS fit on the whole sample (statsmodels OLS).
    Reports each predictor's coefficient + t-stat, R2 and the in-sample fit
    quality (RMSE/MAE/Cor/HitRate).

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

MODE      = "oos"     # "oos" or "insample"

ROLLING   = True      # (oos) rolling vs expanding window
TRAIN_OBS = 60        # (oos) in-sample window length
TIME_LAG  = 1

# Leave empty to use ALL predictors, or list column names to use a subset
# (kept in the order given), e.g. USE_PREDICTORS = ["Unemployment_diff1_lag1"].
USE_PREDICTORS = []
# =========================================================================


def prepare():
    dates, y, X_raw, pred_names = util.load_data(CSV_PATH)
    if USE_PREDICTORS:
        missing = [p for p in USE_PREDICTORS if p not in pred_names]
        if missing:
            raise ValueError(f"Predictor(s) not found in CSV: {missing}")
        cols = [pred_names.index(p) for p in USE_PREDICTORS]
        X_raw = X_raw[:, cols]
        pred_names = list(USE_PREDICTORS)
    print(f"Loaded {len(y)} observations; using {X_raw.shape[1]} predictors from {CSV_PATH}")
    print("Predictors: " + ", ".join(pred_names))
    X_lag = util.apply_lag(X_raw, TIME_LAG)
    return dates, y, X_lag, pred_names


def run_oos():
    dates, y, X_lag, pred_names = prepare()
    yhat, yhat_bm = util.rolling_oos_forecast(y, X_lag, TRAIN_OBS, ROLLING)
    oos = util.evaluate_oos(y, yhat_bm, yhat)
    fq = util.forecast_quality(y, yhat)
    valid = ~(np.isnan(y) | np.isnan(yhat))
    beg, end = util.oos_date_range(dates, valid)

    print("\n========== Full model: OOS metrics ==========")
    print(f"  Predictors : {len(pred_names)}")
    print(f"  OOS obs    : {int(valid.sum())}")
    print(f"  OOS R2     : {oos['R2OOS']:.6f}")
    print(f"  CW stat/p  : {oos['CW']:.4f} / {oos['CWp']:.4f}")
    print(f"  DM stat/p  : {oos['DM']:.4f} / {oos['DMp']:.4f}")
    print(f"  RMSE / MAE : {fq['RMSE']:.6f} / {fq['MAE']:.6f}")
    print(f"  Cor/Hit    : {fq['Cor']:.6f} / {fq['HitRate']:.6f}")
    print(f"  MZ R2/F/p  : {fq['MZ_R2']:.4f} / {fq['MZ_F']:.4f} / {fq['MZ_p']:.4f}")
    print(f"  OOS period : {beg} to {end}")

    summary = {
        "Mode": "oos", "WindowType": "rolling" if ROLLING else "expanding",
        "TrainObs": TRAIN_OBS, "TimeLag": TIME_LAG, "NumPredictors": len(pred_names),
        "OOS_Beg": beg, "OOS_End": end, "Num_OOS_Obs": int(valid.sum()),
        "R2_OOS": oos["R2OOS"], "CW_stat": oos["CW"], "CW_p": oos["CWp"],
        "DM_stat": oos["DM"], "DM_p": oos["DMp"],
        "RMSE": fq["RMSE"], "MAE": fq["MAE"], "Cor": fq["Cor"], "HitRate": fq["HitRate"],
        "R2_MZ": fq["MZ_R2"], "F_MZ": fq["MZ_F"], "p_MZ": fq["MZ_p"],
    }
    util.ensure_dir(OUTPUT_DIR)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    pred_path = os.path.join(OUTPUT_DIR, f"full_oos_predictions_{ts}.csv")
    pd.DataFrame({"Date": dates, "Actual": y,
                  "Forecast": yhat, "Benchmark": yhat_bm}).to_csv(pred_path, index=False)
    sum_path, _ = util.save_summary(OUTPUT_DIR, "full_oos", summary)
    print(f"\nSummary saved to:     {sum_path}")
    print(f"Predictions saved to: {pred_path}")


def run_insample():
    _, y, X_lag, pred_names = prepare()
    m = ~(np.isnan(y) | np.isnan(X_lag).any(axis=1))
    res = sm.OLS(y[m], sm.add_constant(X_lag[m, :])).fit()
    yhat = np.asarray(res.fittedvalues)
    fq = util.forecast_quality(y[m], yhat)

    print(f"\n========== Full model: in-sample OLS ({int(m.sum())} obs) ==========")
    print(f"  Intercept : {res.params[0]:+.4g} (t = {res.tvalues[0]:6.2f})")
    for j, name in enumerate(pred_names):
        print(f"  {name:<32} {res.params[j + 1]:+.4g} (t = {res.tvalues[j + 1]:6.2f})")
    print(f"  R2        : {res.rsquared:.6f}")
    print(f"  RMSE / MAE: {fq['RMSE']:.6f} / {fq['MAE']:.6f}   Cor: {fq['Cor']:.4f}")

    coef = pd.DataFrame({
        "Term": ["Intercept"] + pred_names,
        "Coef": res.params, "t_stat": res.tvalues, "p_value": res.pvalues,
    })
    util.ensure_dir(OUTPUT_DIR)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    coef_path = os.path.join(OUTPUT_DIR, f"full_insample_coefs_{ts}.csv")
    coef.to_csv(coef_path, index=False)

    summary = {
        "Mode": "insample", "TimeLag": TIME_LAG, "NumPredictors": len(pred_names),
        "Num_Obs": int(m.sum()), "R2": res.rsquared, "Adj_R2": res.rsquared_adj,
        "F_stat": res.fvalue, "F_p": res.f_pvalue, "AIC": res.aic, "BIC": res.bic,
        "RMSE": fq["RMSE"], "MAE": fq["MAE"], "Cor": fq["Cor"], "HitRate": fq["HitRate"],
    }
    sum_path, _ = util.save_summary(OUTPUT_DIR, "full_insample", summary)
    print(f"\nCoefficients saved to: {coef_path}")
    print(f"Summary saved to:      {sum_path}")


def main():
    if MODE == "oos":
        run_oos()
    elif MODE == "insample":
        run_insample()
    else:
        raise ValueError(f"Unknown MODE: {MODE}")


if __name__ == "__main__":
    main()
