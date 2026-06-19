"""
inflation_ar.py
Pure autoregressive one-step-ahead OOS forecast. The target is predicted from
its OWN past values only, with a reporting (publication) lag:

    regressors for y(t):  y(t-(r+1)), y(t-(r+2)), ..., y(t-(r+p))
        r = REPORT_LAG   (0 -> uses t-1, t-2, ...; 1 -> uses t-2, t-3, ...)
        p = LOOKBACK     (number of AR lags)

Optionally p is chosen at each origin by AIC over LOOKBACK_GRID.

CSV format (1 data column + date): col 0 = date, col 1 = target.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import util

# =========================================================================
#  CONFIG — edit here
# =========================================================================
CSV_PATH   = "./DATA/Liedtke/US/aggregated.csv"   # uses col 0 = date, col 1 = target
OUTPUT_DIR = "./RESULTS/"

ROLLING       = True
TRAIN_OBS     = 60
REPORT_LAG    = 1          # r: first usable lag is y(t-(r+1))
LOOKBACK      = 1          # p: number of AR lags (if not optimal)

OPTIMAL_LOOKBACK = False   # True = pick p by AIC at each origin
LOOKBACK_GRID    = range(1, 7)
# =========================================================================


def lag_matrix(y, lags):
    """Column k = y shifted down by lags[k] (NaN at the top)."""
    n = len(y)
    M = np.full((n, len(lags)), np.nan)
    for k, L in enumerate(lags):
        if L < n:
            M[L:, k] = y[: n - L]
    return M


# =========================================================================
#  Load + build regressors
# =========================================================================
df = pd.read_csv(CSV_PATH)
dates = pd.to_datetime(df.iloc[:, 0])
y = df.iloc[:, 1].to_numpy(dtype=float)
n = len(y)

p_max = max(LOOKBACK_GRID) if OPTIMAL_LOOKBACK else LOOKBACK
# regressor column k (0-based) corresponds to lag (REPORT_LAG + 1 + k)
reg_lags = [REPORT_LAG + 1 + k for k in range(p_max)]
RegMax = lag_matrix(y, reg_lags)

print(f"Loaded {n} observations from {CSV_PATH}")
print(f"Reporting lag r = {REPORT_LAG}  ->  first usable lag is y(t-{REPORT_LAG + 1})")
if OPTIMAL_LOOKBACK:
    print(f"Lookback: optimal by AIC over {list(LOOKBACK_GRID)}")
else:
    print(f"Lookback p = {LOOKBACK}")

# =========================================================================
#  Rolling one-step-ahead loop
# =========================================================================
yhat = np.full(n, np.nan)
yhat_bm = np.full(n, np.nan)
lag_used = np.full(n, np.nan)

for t in range(TRAIN_OBS - 1, n - 1):
    win = util.window_index(t, TRAIN_OBS, ROLLING)
    i_out = t + 1
    if np.isnan(y[win]).any():
        continue

    if OPTIMAL_LOOKBACK:
        best_aic, best_p = np.inf, None
        for p in LOOKBACK_GRID:
            Xin = RegMax[win, :p]
            xout = RegMax[i_out, :p]
            if np.isnan(Xin).any() or np.isnan(xout).any():
                continue
            aic = util.info_criterion(y, RegMax[:, :p], win, which="AIC")
            if aic < best_aic:
                best_aic, best_p = aic, p
        if best_p is None:
            continue
        p = best_p
    else:
        p = LOOKBACK

    Xin = RegMax[win, :p]
    xout = RegMax[i_out, :p]
    if np.isnan(Xin).any() or np.isnan(xout).any():
        continue

    model = LinearRegression().fit(Xin, y[win])
    yhat[i_out] = model.predict(xout.reshape(1, -1))[0]
    yhat_bm[i_out] = y[win].mean()
    lag_used[i_out] = p

# =========================================================================
#  Metrics
# =========================================================================
oos = util.evaluate_oos(y, yhat_bm, yhat)
fq = util.forecast_quality(y, yhat)
valid = ~(np.isnan(y) | np.isnan(yhat))
beg = dates[valid].iloc[0].strftime("%Y-%m-%d") if valid.any() else ""
end = dates[valid].iloc[-1].strftime("%Y-%m-%d") if valid.any() else ""

print("\n========== AR model metrics ==========")
print(f"  OOS obs    : {int(valid.sum())}")
if OPTIMAL_LOOKBACK and valid.any():
    print(f"  Lookback used: {int(np.nanmin(lag_used))} to {int(np.nanmax(lag_used))} "
          f"(median {np.nanmedian(lag_used):g})")
print(f"  OOS R2     : {oos['R2OOS']:.6f}   OOS R2-CT: {oos['R2OOS_CT']:.6f}")
print(f"  CW  / p    : {oos['CW']:.4f} / {oos['CWp']:.4f}")
print(f"  DM  / p    : {oos['DM']:.4f} / {oos['DMp']:.4f}")
print(f"  RMSE / MAE : {fq['RMSE']:.6f} / {fq['MAE']:.6f}")
print(f"  Cor / Hit  : {fq['Cor']:.6f} / {fq['HitRate']:.6f}")

# =========================================================================
#  Save
# =========================================================================
summary = {
    "WindowType": "rolling" if ROLLING else "expanding",
    "TrainObs": TRAIN_OBS, "ReportLag": REPORT_LAG,
    "Lookback": f"optimal{list(LOOKBACK_GRID)}" if OPTIMAL_LOOKBACK else LOOKBACK,
    "OOS_Beg": beg, "OOS_End": end, "Num_OOS_Obs": int(valid.sum()),
    "R2_OOS": oos["R2OOS"], "R2_OOS_CT": oos["R2OOS_CT"],
    "CW_stat": oos["CW"], "CW_p": oos["CWp"],
    "CW_stat_CT": oos["CW_CT"], "CW_p_CT": oos["CWp_CT"],
    "DM_stat": oos["DM"], "DM_p": oos["DMp"],
    "RMSE": fq["RMSE"], "MAE": fq["MAE"], "Cor": fq["Cor"], "HitRate": fq["HitRate"],
    "R2_MZ": fq["MZ_R2"], "F_MZ": fq["MZ_F"], "p_MZ": fq["MZ_p"],
}
sum_path, ts = util.save_summary(OUTPUT_DIR, "ar", summary)

pred_path = os.path.join(OUTPUT_DIR, f"ar_predictions_{ts}.csv")
pd.DataFrame({"Date": dates, "Actual": y, "Forecast": yhat,
              "Benchmark": yhat_bm, "LookbackUsed": lag_used}
             ).to_csv(pred_path, index=False)
print(f"\nSummary saved to:     {sum_path}")
print(f"Predictions saved to: {pred_path}")
