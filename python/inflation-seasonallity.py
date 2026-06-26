"""
inflation-seasonallity.py

Rolling/expanding one-step-ahead inflation forecast using seasonal dummies
only. Month 1 is the reference category, so the regression uses 11 dummies
for months 2..12 plus an intercept.

The script follows the same data loading and rolling window settings as the
other inflation forecasting scripts in this folder.
"""

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import util

# =========================================================================
#  CONFIG
# =========================================================================
COUNTRY = util.cfg("COUNTRY", "UK")
CSV_PATH = f"./DATA/Liedtke/{COUNTRY}/aggregated.csv"
OUTPUT_DIR = "./RESULTS/"

ROLLING = util.cfg("ROLLING", True)
TRAIN_OBS = int(util.cfg("TRAIN_OBS", 120))
TIME_LAG = 1
# =========================================================================


def prepare():
    dates, y, _, _ = util.load_data(CSV_PATH)
    months = pd.to_datetime(dates).dt.month.to_numpy(dtype=int)
    print(f"Loaded {len(y)} observations from {CSV_PATH}")
    return dates.reset_index(drop=True), y, months


def month_dummies(months):
    """11 seasonal dummies using month 1 as the reference category."""
    dummies = np.zeros((len(months), 11), dtype=float)
    for j in range(11):
        dummies[:, j] = (months == j + 2).astype(float)
    return dummies


def run_oos():
    dates, y, months = prepare()
    n = len(y)

    yhat = np.full(n, np.nan)
    yhat_bm = np.full(n, np.nan)

    for t in range(TRAIN_OBS - 1, n - 1):
        idx = util.window_index(t, TRAIN_OBS, ROLLING)
        i_out = t + 1

        yin = y[idx]
        if np.isnan(yin).any() or np.isnan(y[i_out]):
            continue

        X_in = month_dummies(months[idx])
        X_out = month_dummies(np.array([months[i_out]]))

        model = LinearRegression().fit(X_in, yin)
        yhat[i_out] = float(model.predict(X_out)[0])
        yhat_bm[i_out] = float(yin.mean())

    oos = util.evaluate_oos(y, yhat_bm, yhat)
    fq = util.forecast_quality(y, yhat)
    valid = ~(np.isnan(y) | np.isnan(yhat))
    beg, end = util.oos_date_range(dates, valid)

    print("\n========== Seasonal dummies only: OOS metrics ==========")
    print(f"  OOS obs    : {int(valid.sum())}")
    print(f"  OOS R2     : {oos['R2OOS']:.6f}")
    print(f"  CW stat/p  : {oos['CW']:.4f} / {oos['CWp']:.4f}")
    print(f"  DM stat/p  : {oos['DM']:.4f} / {oos['DMp']:.4f}")
    print(f"  RMSE / MAE : {fq['RMSE']:.6f} / {fq['MAE']:.6f}")
    print(f"  Cor/Hit    : {fq['Cor']:.6f} / {fq['HitRate']:.6f}")
    print(f"  OOS period : {beg} to {end}")

    summary = {
        "Mode": "oos",
        "WindowType": "rolling" if ROLLING else "expanding",
        "TrainObs": TRAIN_OBS,
        "TimeLag": TIME_LAG,
        "NumPredictors": 11,
        "OOS_Beg": beg,
        "OOS_End": end,
        "Num_OOS_Obs": int(valid.sum()),
        "R2_OOS": oos["R2OOS"],
        "CW_stat": oos["CW"],
        "CW_p": oos["CWp"],
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

    options = f"train{TRAIN_OBS}_{util.window_tag(ROLLING)}_seasonal_dummies"
    out_dir = util.result_dir(OUTPUT_DIR, "seasonality", COUNTRY, "oos", options)
    pred_path = os.path.join(out_dir, "predictions.csv")
    pd.DataFrame({
        "Date": dates,
        "Actual": y,
        "Forecast": yhat,
        "Benchmark": yhat_bm,
    }).to_csv(pred_path, index=False)
    sum_path, _ = util.save_summary(out_dir, "seasonality_oos", summary)

    print(f"\nSummary saved to:     {sum_path}")
    print(f"Predictions saved to: {pred_path}")


def main():
    run_oos()


if __name__ == "__main__":
    main()