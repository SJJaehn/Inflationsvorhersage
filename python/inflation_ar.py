"""
inflation_ar.py
Autoregressive one-step-ahead OOS forecast.

AR mode (VAR_MODE = False):
    The target is predicted from its OWN past values only, with a reporting
    (publication) lag:

        regressors for y(t):  y(t-(r+1)), y(t-(r+2)), ..., y(t-(r+p))
            r = REPORT_LAG   (0 -> uses t-1, t-2, ...; 1 -> uses t-2, t-3, ...)
            p = LOOKBACK     (number of AR lags)

    Optionally p is chosen at each origin by AIC over LOOKBACK_GRID.

VARX mode (VAR_MODE = True):
    The AR lag terms above PLUS the macro predictors (cols 2..) as exogenous
    regressors. Only the TARGET needs a reporting lag here: the predictors are
    already reporting-lag aligned in the CSV, so they enter contemporaneously
    (the row aligned with y(t)). Predictors that are numerically collinear with
    an AR lag term (e.g. a lagged copy of the target's own series) are dropped
    automatically so the design does not become rank deficient.

CSV format: col 0 = date, col 1 = target, col 2.. = predictors (VARX only).
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import util

# =========================================================================
#  CONFIG — edit here
# =========================================================================
CSV_PATH   = "./DATA/Liedtke/US/aggregated.csv"   # col 0 = date, col 1 = target, col 2.. = predictors
OUTPUT_DIR = "./RESULTS/"

VAR_MODE      = False     # False = AR (target lags only), True = VARX (+ predictors)

ROLLING       = False     # MATLAB bKap4_3 uses expanding window (lRoll=false)
TRAIN_OBS     = 239       # matches MATLAB: rolling window is (t-iNumIn+1):(t-1) = iNumIn-1 obs
REPORT_LAG    = 1          # r: first usable lag is y(t-(r+1)) (applies to the target only)
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

# Exogenous predictors (VARX only). They are already reporting-lag aligned in
# the CSV, so they enter contemporaneously (no extra shift).
Xpred = np.empty((n, 0))
pred_names = []
if VAR_MODE:
    Xpred = df.iloc[:, 2:].to_numpy(dtype=float)
    pred_names = list(df.columns[2:])

    # Drop predictors numerically collinear with any AR lag term (data-driven,
    # name-agnostic): a predictor identical to y(t-(r+1+k)) makes the design
    # rank deficient. Detect by (near-)perfect correlation over complete rows.
    keep = np.ones(Xpred.shape[1], dtype=bool)
    for j in range(Xpred.shape[1]):
        for k in range(p_max):
            ok = ~(np.isnan(Xpred[:, j]) | np.isnan(RegMax[:, k]))
            if ok.sum() > 2:
                a = Xpred[ok, j] - Xpred[ok, j].mean()
                b = RegMax[ok, k] - RegMax[ok, k].mean()
                denom = np.sqrt((a @ a) * (b @ b))
                if denom > 0 and abs(a @ b) / denom >= 1 - 1e-8:
                    keep[j] = False
                    break
    dropped = [pred_names[j] for j in range(len(pred_names)) if not keep[j]]
    Xpred = Xpred[:, keep]
    pred_names = [pred_names[j] for j in range(len(pred_names)) if keep[j]]
    if dropped:
        print(f"Dropped {len(dropped)} predictor(s) collinear with the AR lag: "
              f"{', '.join(dropped)}")

print(f"Loaded {n} observations from {CSV_PATH}")
print(f"Model: {'VARX (AR lags + predictors)' if VAR_MODE else 'AR (target lags only)'}")
print(f"Reporting lag r = {REPORT_LAG}  ->  first usable lag is y(t-{REPORT_LAG + 1})")
if VAR_MODE:
    print(f"Exogenous predictors: {Xpred.shape[1]}")
if OPTIMAL_LOOKBACK:
    print(f"Lookback: optimal by AIC over {list(LOOKBACK_GRID)}")
else:
    print(f"Lookback p = {LOOKBACK}")

# =========================================================================
#  Compact to complete cases (mirrors MATLAB: NaN rows stripped upfront so
#  expanding windows don't fail on early NaN).  For pure AR the complete-case
#  mask uses y and the maximum AR-lag column; for VARX it also requires all
#  exogenous predictors to be present.
# =========================================================================
all_regs = RegMax[:, :p_max] if Xpred.shape[1] == 0 else np.column_stack(
    [RegMax[:, :p_max], Xpred])
ok = ~(np.isnan(y) | np.isnan(all_regs).any(axis=1))
orig_idx = np.where(ok)[0]      # original positions of compact rows
yc       = y[ok]
RegMaxC  = RegMax[ok, :]
XpredC   = Xpred[ok, :] if Xpred.shape[1] else Xpred
nc       = len(yc)

# =========================================================================
#  Rolling one-step-ahead loop (operates on the compact NaN-free series)
# =========================================================================
yhat    = np.full(n, np.nan)
yhat_bm = np.full(n, np.nan)
yhat_lv = np.full(n, np.nan)
lag_used = np.full(n, np.nan)

for t in range(TRAIN_OBS - 1, nc - 1):
    win   = util.window_index(t, TRAIN_OBS, ROLLING)
    i_out = t + 1
    orig_out = orig_idx[i_out]

    if OPTIMAL_LOOKBACK:
        best_aic, best_p = np.inf, None
        for p in LOOKBACK_GRID:
            Reg = np.column_stack([RegMaxC[:, :p], XpredC]) if XpredC.shape[1] else RegMaxC[:, :p]
            aic = util.info_criterion(yc, Reg, win, which="AIC")
            if aic < best_aic:
                best_aic, best_p = aic, p
        if best_p is None:
            continue
        p = best_p
    else:
        p = LOOKBACK

    Xin  = np.column_stack([RegMaxC[win, :p], XpredC[win]]) if XpredC.shape[1] else RegMaxC[win, :p]
    xout = np.concatenate([RegMaxC[i_out, :p], XpredC[i_out]]) if XpredC.shape[1] else RegMaxC[i_out, :p]

    model = LinearRegression().fit(Xin, yc[win])
    yhat[orig_out]    = model.predict(xout.reshape(1, -1))[0]
    yhat_bm[orig_out] = yc[win].mean()
    lag_used[orig_out] = p

    # Last-value benchmark: y(t-r) in original indices
    lv_orig = orig_out - 1 - REPORT_LAG
    if lv_orig >= 0 and not np.isnan(y[lv_orig]):
        yhat_lv[orig_out] = y[lv_orig]

# =========================================================================
#  Metrics
# =========================================================================
oos = util.evaluate_oos(y, yhat_bm, yhat)
fq = util.forecast_quality(y, yhat)
oos_lv = util.evaluate_oos(y, yhat_bm, yhat_lv)   # last-value vs historical mean
fq_lv = util.forecast_quality(y, yhat_lv)
valid = ~(np.isnan(y) | np.isnan(yhat))
beg = dates[valid].iloc[0].strftime("%Y-%m-%d") if valid.any() else ""
end = dates[valid].iloc[-1].strftime("%Y-%m-%d") if valid.any() else ""

print(f"\n========== {'VARX' if VAR_MODE else 'AR'} model metrics ==========")
print(f"  OOS obs    : {int(valid.sum())}")
if OPTIMAL_LOOKBACK and valid.any():
    print(f"  Lookback used: {int(np.nanmin(lag_used))} to {int(np.nanmax(lag_used))} "
          f"(median {np.nanmedian(lag_used):g})")
print(f"  OOS R2     : {oos['R2OOS']:.6f}   OOS R2-CT: {oos['R2OOS_CT']:.6f}")
print(f"  CW  / p    : {oos['CW']:.4f} / {oos['CWp']:.4f}")
print(f"  DM  / p    : {oos['DM']:.4f} / {oos['DMp']:.4f}")
print(f"  RMSE / MAE : {fq['RMSE']:.6f} / {fq['MAE']:.6f}")
print(f"  Cor / Hit  : {fq['Cor']:.6f} / {fq['HitRate']:.6f}")

print(f"\n========== Last-value benchmark [yhat(t+1)=y(t-{REPORT_LAG})] ==========")
print(f"  OOS R2     : {oos_lv['R2OOS']:.6f}   OOS R2-CT: {oos_lv['R2OOS_CT']:.6f}")
print(f"  RMSE / MAE : {fq_lv['RMSE']:.6f} / {fq_lv['MAE']:.6f}")
print(f"  Cor / Hit  : {fq_lv['Cor']:.6f} / {fq_lv['HitRate']:.6f}")

# =========================================================================
#  Save
# =========================================================================
summary = {
    "Model": "VARX" if VAR_MODE else "AR",
    "NumPredictors": Xpred.shape[1],
    "Predictors": "; ".join(pred_names),
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
    "LV_R2_OOS": oos_lv["R2OOS"], "LV_R2_OOS_CT": oos_lv["R2OOS_CT"],
    "LV_RMSE": fq_lv["RMSE"], "LV_MAE": fq_lv["MAE"],
    "LV_Cor": fq_lv["Cor"], "LV_HitRate": fq_lv["HitRate"],
}
prefix = "varx" if VAR_MODE else "ar"
sum_path, ts = util.save_summary(OUTPUT_DIR, prefix, summary)

pred_path = os.path.join(OUTPUT_DIR, f"{prefix}_predictions_{ts}.csv")
pd.DataFrame({"Date": dates, "Actual": y, "Forecast": yhat,
              "Benchmark": yhat_bm, "LastValue": yhat_lv, "LookbackUsed": lag_used}
             ).to_csv(pred_path, index=False)
print(f"\nSummary saved to:     {sum_path}")
print(f"Predictions saved to: {pred_path}")
