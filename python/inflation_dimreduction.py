"""
inflation_dimreduction.py
Dimensionality-reduction one-step-ahead OOS forecast of inflation, mirroring the
MATLAB bKap5 demos but built on sklearn.

Two methods (set MODE below):
  'PCA' : principal-component regression (PCR). The macro panel is reduced to a
          few principal components (unsupervised, target-blind) and inflation is
          regressed on those components.
  'PLS' : partial-least-squares regression. The components are extracted to
          maximise covariance with the target (supervised), then used to predict.

At each rolling/expanding origin the reduction AND the regression are estimated
on the in-sample window only, and the next-step predictor row is projected with
those in-sample parameters (standardisation, eigenvectors / PLS weights, betas),
so there is no look-ahead. Forecasts are compared against the historical-mean
benchmark.

CSV format: col 0 = date, col 1 = target, col 2.. = predictors (already
reporting-lag aligned in DATA/Liedtke/aggregate.py).
"""

import os
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import util

# =========================================================================
#  CONFIG — edit here
# =========================================================================
CSV_PATH = "./DATA/Liedtke/US/aggregated.csv"
OUTPUT_DIR = "./RESULTS/"

MODE = "PCA"  # 'PCA' (unsupervised PCR) or 'PLS' (supervised)
ROLLING = False  # MATLAB bKap5_5/bKap5_6 use expanding window (lRoll=false)
TRAIN_OBS = 239  # matches MATLAB: rolling window is (t-iNumIn+1):(t-1) = iNumIn-1 obs
TIME_LAG = 1  # additional predictive lag on predictors (1 = standard)
NUM_COMP = 3  # number of components to retain
STANDARDIZE = True  # z-standardise predictors (they live on different scales)
# =========================================================================

if MODE not in ("PCA", "PLS"):
    raise ValueError(f"Unknown MODE: {MODE!r} (use 'PCA' or 'PLS')")

dates, y, X_raw, pred_names = util.load_data(CSV_PATH)
n = len(y)
X_lag = util.apply_lag(X_raw, TIME_LAG)
num_comp = min(NUM_COMP, X_lag.shape[1])

print(f"Loaded {n} observations, {X_lag.shape[1]} predictors from {CSV_PATH}")
print(f"Method: {MODE} | components: {num_comp} | "
      f"window: {'rolling' if ROLLING else 'expanding'} (train {TRAIN_OBS})")


def fit_predict(Xin, yin, xout):
    """Fit the reduction + regression on the in-sample window and return the
    one-step-ahead forecast for the (1 x k) out-of-sample predictor row."""
    if STANDARDIZE:
        scaler = StandardScaler().fit(Xin)
        Xin = scaler.transform(Xin)
        xout = scaler.transform(xout)

    if MODE == "PCA":
        # Unsupervised reduction, then OLS of y on the in-sample scores.
        pca = PCA(n_components=num_comp).fit(Xin)
        scores_in = pca.transform(Xin)
        scores_out = pca.transform(xout)
        reg = LinearRegression().fit(scores_in, yin)
        return reg.predict(scores_out)[0]

    # PLS: supervised; PLSRegression bundles reduction + regression.
    pls = PLSRegression(n_components=num_comp, scale=False).fit(Xin, yin)
    return pls.predict(xout).ravel()[0]


# Compact to complete cases upfront (mirrors MATLAB's upfront NaN removal so
# that expanding windows start from the first complete observation).
ok       = ~(np.isnan(y) | np.isnan(X_lag).any(axis=1))
orig_idx = np.where(ok)[0]
yc       = y[ok]
Xc       = X_lag[ok, :]
nc       = len(yc)

# =========================================================================
#  Rolling one-step-ahead loop (operates on the compact NaN-free series)
# =========================================================================
yhat    = np.full(n, np.nan)
yhat_bm = np.full(n, np.nan)

for t in range(TRAIN_OBS - 1, nc - 1):
    win      = util.window_index(t, TRAIN_OBS, ROLLING)
    i_out    = t + 1
    orig_out = orig_idx[i_out]

    yin  = yc[win]
    Xin  = Xc[win, :]
    xout = Xc[i_out, :]

    yhat[orig_out]    = fit_predict(Xin, yin, xout.reshape(1, -1))
    yhat_bm[orig_out] = yin.mean()

# =========================================================================
#  Metrics
# =========================================================================
oos = util.evaluate_oos(y, yhat_bm, yhat)
fq = util.forecast_quality(y, yhat)
valid = ~(np.isnan(y) | np.isnan(yhat))
beg = dates[valid].iloc[0].strftime("%Y-%m-%d") if valid.any() else ""
end = dates[valid].iloc[-1].strftime("%Y-%m-%d") if valid.any() else ""

print(f"\n========== {MODE} model metrics ==========")
print(f"  Components : {num_comp}")
print(f"  OOS obs    : {int(valid.sum())}")
print(f"  OOS R2     : {oos['R2OOS']:.6f}   OOS R2-CT: {oos['R2OOS_CT']:.6f}")
print(f"  CW  / p    : {oos['CW']:.4f} / {oos['CWp']:.4f}")
print(f"  DM  / p    : {oos['DM']:.4f} / {oos['DMp']:.4f}")
print(f"  RMSE / MAE : {fq['RMSE']:.6f} / {fq['MAE']:.6f}")
print(f"  Cor / Hit  : {fq['Cor']:.6f} / {fq['HitRate']:.6f}")

# =========================================================================
#  Save
# =========================================================================
summary = {
    "Method": MODE,
    "NumComponents": num_comp,
    "Standardize": STANDARDIZE,
    "WindowType": "rolling" if ROLLING else "expanding",
    "TrainObs": TRAIN_OBS,
    "TimeLag": TIME_LAG,
    "NumPredictors": X_lag.shape[1],
    "OOS_Beg": beg, "OOS_End": end, "Num_OOS_Obs": int(valid.sum()),
    "R2_OOS": oos["R2OOS"], "R2_OOS_CT": oos["R2OOS_CT"],
    "CW_stat": oos["CW"], "CW_p": oos["CWp"],
    "CW_stat_CT": oos["CW_CT"], "CW_p_CT": oos["CWp_CT"],
    "DM_stat": oos["DM"], "DM_p": oos["DMp"],
    "RMSE": fq["RMSE"], "MAE": fq["MAE"], "Cor": fq["Cor"], "HitRate": fq["HitRate"],
    "R2_MZ": fq["MZ_R2"], "F_MZ": fq["MZ_F"], "p_MZ": fq["MZ_p"],
}
prefix = MODE.lower()
sum_path, ts = util.save_summary(OUTPUT_DIR, prefix, summary)

pred_path = os.path.join(OUTPUT_DIR, f"{prefix}_predictions_{ts}.csv")
pd.DataFrame({"Date": dates, "Actual": y, "Forecast": yhat, "Benchmark": yhat_bm}
             ).to_csv(pred_path, index=False)
print(f"\nSummary saved to:     {sum_path}")
print(f"Predictions saved to: {pred_path}")
