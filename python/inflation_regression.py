"""
inflation_regression.py
Rolling/expanding one-step-ahead OOS predictive regression.

Two modes (set MODE below):
  'single' : loop over predictors, one univariate regression each -> a table
             with one row of metrics per predictor.
  'full'   : one "kitchen-sink" regression using ALL predictors (or the subset
             in USE_PREDICTORS) at once -> a single summary + prediction file.

CSV format: col 0 = date, col 1 = target, col 2.. = predictors.
"""

import os

import numpy as np
import pandas as pd

import util

# =========================================================================
#  CONFIG — edit here
# =========================================================================
CSV_PATH = "./DATA/Liedtke/US/aggregated.csv"
OUTPUT_DIR = "./RESULTS/"

MODE = "single"  # 'single' or 'full'
ROLLING = True  # True = rolling window, False = expanding
TRAIN_OBS = 120  # in-sample window length
TIME_LAG = 1  # predictor lag (1 = standard predictive regression)

# Single mode only: if True, evaluate every predictor on the SAME sample (rows
# where the target AND all predictors exist), so the R2s are comparable across
# predictors (and to the selection script).
SHARED_TIMEFRAME = False

# Only for MODE == 'full'. Empty list = use all predictors. Otherwise a list
# of predictor names, kept in the order given.
USE_PREDICTORS = []
# =========================================================================

dates, y, X_raw, pred_names = util.load_data(CSV_PATH)
n_obs = len(y)
print(f"Loaded {n_obs} observations, {X_raw.shape[1]} predictors from {CSV_PATH}")

X_lag = util.apply_lag(X_raw, TIME_LAG)
util.ensure_dir(OUTPUT_DIR)

# Optional: restrict to the shared timeframe (rows where the target and ALL
# predictors have data) so every predictor is scored on the same observations.
if SHARED_TIMEFRAME:
    keep = ~(np.isnan(y) | np.isnan(X_lag).any(axis=1))
    y = y[keep]
    X_lag = X_lag[keep, :]
    dates = dates[keep].reset_index(drop=True)
    n_obs = len(y)
    print(f"Shared timeframe: {n_obs} of {len(keep)} observations retained.")


# =========================================================================
#  MODE 'single': one univariate regression per predictor
# =========================================================================
if MODE == "single":
    rows = []
    for j, name in enumerate(pred_names):
        Xj = X_lag[:, [j]]
        yhat, yhat_bm = util.rolling_oos_forecast(y, Xj, TRAIN_OBS, ROLLING)

        oos = util.evaluate_oos(y, yhat_bm, yhat)
        fq = util.forecast_quality(y, yhat)

        valid = ~(np.isnan(y) | np.isnan(yhat))
        if valid.any():
            beg = dates[valid].iloc[0].strftime("%Y-%m-%d")
            end = dates[valid].iloc[-1].strftime("%Y-%m-%d")
        else:
            beg = end = ""

        rows.append(
            dict(
                Predictor=name,
                OOS_Beg=beg,
                OOS_End=end,
                Num_OOS_Obs=int(valid.sum()),
                R2_OOS=oos["R2OOS"],
                R2_OOS_CT=oos["R2OOS_CT"],
                CW_stat=oos["CW"],
                CW_p=oos["CWp"],
                CW_stat_CT=oos["CW_CT"],
                CW_p_CT=oos["CWp_CT"],
                DM_stat=oos["DM"],
                DM_p=oos["DMp"],
                RMSE=fq["RMSE"],
                MAE=fq["MAE"],
                Cor=fq["Cor"],
                HitRate=fq["HitRate"],
                R2_MZ=fq["MZ_R2"],
                F_MZ=fq["MZ_F"],
                p_MZ=fq["MZ_p"],
            )
        )
        print(f"  {name:20s} OOS R2={oos['R2OOS']:.4f}  RMSE={fq['RMSE']:.4f}")

    tbl = pd.DataFrame(rows)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(OUTPUT_DIR, f"single_results_{ts}.csv")
    tbl.to_csv(out_file, index=False)

    print("\n========== Summary ==========")
    print(
        tbl[
            [
                "Predictor",
                "R2_OOS",
                "R2_OOS_CT",
                "CW_p",
                "CW_p_CT",
                "RMSE",
                "MAE",
                "Cor",
            ]
        ].to_string(index=False)
    )
    print(f"\nResults saved to: {out_file}")


# =========================================================================
#  MODE 'full': single regression on all (or selected) predictors
# =========================================================================
elif MODE == "full":
    if USE_PREDICTORS:
        missing = [p for p in USE_PREDICTORS if p not in pred_names]
        if missing:
            raise ValueError(f"Predictor(s) not found: {missing}")
        cols = [pred_names.index(p) for p in USE_PREDICTORS]  # keeps given order
        X_use = X_lag[:, cols]
        used_names = list(USE_PREDICTORS)
    else:
        X_use = X_lag
        used_names = list(pred_names)

    print(f"Using {len(used_names)} predictors: {', '.join(used_names)}")

    yhat, yhat_bm = util.rolling_oos_forecast(y, X_use, TRAIN_OBS, ROLLING)
    oos = util.evaluate_oos(y, yhat_bm, yhat)
    fq = util.forecast_quality(y, yhat)

    valid = ~(np.isnan(y) | np.isnan(yhat))
    beg = dates[valid].iloc[0].strftime("%Y-%m-%d") if valid.any() else ""
    end = dates[valid].iloc[-1].strftime("%Y-%m-%d") if valid.any() else ""

    print("\n========== Full (all-predictor) model ==========")
    print(f"  Predictors : {len(used_names)}")
    print(f"  OOS obs    : {int(valid.sum())}")
    print(f"  OOS R2     : {oos['R2OOS']:.6f}")
    print(f"  OOS R2-CT  : {oos['R2OOS_CT']:.6f}")
    print(f"  CW  / p    : {oos['CW']:.4f} / {oos['CWp']:.4f}")
    print(f"  DM  / p    : {oos['DM']:.4f} / {oos['DMp']:.4f}")
    print(f"  RMSE / MAE : {fq['RMSE']:.6f} / {fq['MAE']:.6f}")
    print(f"  Cor / Hit  : {fq['Cor']:.6f} / {fq['HitRate']:.6f}")

    summary = {
        "WindowType": "rolling" if ROLLING else "expanding",
        "TrainObs": TRAIN_OBS,
        "TimeLag": TIME_LAG,
        "NumPredictors": len(used_names),
        "Predictors": "; ".join(used_names),
        "OOS_Beg": beg,
        "OOS_End": end,
        "Num_OOS_Obs": int(valid.sum()),
        "R2_OOS": oos["R2OOS"],
        "R2_OOS_CT": oos["R2OOS_CT"],
        "CW_stat": oos["CW"],
        "CW_p": oos["CWp"],
        "CW_stat_CT": oos["CW_CT"],
        "CW_p_CT": oos["CWp_CT"],
        "DM_stat": oos["DM"],
        "DM_p": oos["DMp"],
        "RMSE": fq["RMSE"],
        "MAE": fq["MAE"],
        "Cor": fq["Cor"],
        "HitRate": fq["HitRate"],
        "R2_MZ": fq["MZ_R2"],
        "F_MZ": fq["MZ_F"],
        "p_MZ": fq["MZ_p"],
    }
    sum_path, ts = util.save_summary(OUTPUT_DIR, "full", summary)

    pred_path = os.path.join(OUTPUT_DIR, f"full_predictions_{ts}.csv")
    pd.DataFrame(
        {"Date": dates, "Actual": y, "Forecast": yhat, "Benchmark": yhat_bm}
    ).to_csv(pred_path, index=False)
    print(f"\nSummary saved to:     {sum_path}")
    print(f"Predictions saved to: {pred_path}")

else:
    raise ValueError(f"Unknown MODE: {MODE!r} (use 'single' or 'full')")
