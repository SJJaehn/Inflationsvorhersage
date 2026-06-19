"""
inflation_regression_insample.py
In-sample predictive regressions of inflation (counterpart to the OOS
inflation_regression.py; mirrors MATLAB bKap3_3 / bKap3_5).

Two modes (set MODE below):
  'single' : one univariate regression per predictor -> a table with the
             standardised slope, its t-stat and R2 for each predictor.
  'full'   : one "kitchen-sink" regression on ALL predictors at once -> the
             standardised coefficients, t-stats and the joint R2.

Predictors are lagged one period (they are already reporting-lag aligned in the
CSV) and z-standardised, so the coefficients are comparable across predictors.
The fits use statsmodels OLS.

CSV format: col 0 = date, col 1 = target, col 2.. = predictors.
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

import util

# =========================================================================
#  CONFIG — edit here
# =========================================================================
CSV_PATH = "./DATA/Liedtke/US/aggregated.csv"
OUTPUT_DIR = "./RESULTS/"

MODE = "single"  # 'single' (one regression per predictor) or 'full' (all at once)
TIME_LAG = 1     # additional predictive lag (1 = standard predictive regression)
# =========================================================================

if MODE not in ("single", "full"):
    raise ValueError(f"Unknown MODE: {MODE!r} (use 'single' or 'full')")

dates, y, X_raw, pred_names = util.load_data(CSV_PATH)
X_lag = util.apply_lag(X_raw, TIME_LAG)
util.ensure_dir(OUTPUT_DIR)
print(f"Loaded {len(y)} observations, {X_lag.shape[1]} predictors from {CSV_PATH}")


def zscore_fit(Xc):
    """z-standardise complete-case columns with sklearn StandardScaler."""
    return StandardScaler().fit_transform(Xc)


# =========================================================================
#  MODE 'single': one univariate regression per predictor
# =========================================================================
if MODE == "single":
    rows = []
    for j, name in enumerate(pred_names):
        keep = ~(np.isnan(y) | np.isnan(X_lag[:, j]))
        if keep.sum() < 3:
            continue
        xz = zscore_fit(X_lag[keep, j].reshape(-1, 1))
        res = sm.OLS(y[keep], sm.add_constant(xz)).fit()
        rows.append(dict(
            Predictor=name,
            Beg=dates[keep].iloc[0].strftime("%Y-%m-%d"),
            End=dates[keep].iloc[-1].strftime("%Y-%m-%d"),
            NumObs=int(keep.sum()),
            Beta=res.params[1],
            Beta_t=res.tvalues[1],
            R2=res.rsquared,
        ))
        print(f"  {name:22s} beta={res.params[1]:+.4f} (t={res.tvalues[1]:6.2f})  R2={res.rsquared:.4f}")

    tbl = pd.DataFrame(rows)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(OUTPUT_DIR, f"insample_single_{ts}.csv")
    tbl.to_csv(out_file, index=False)
    print("\n========== In-sample single-predictor regressions ==========")
    print(tbl[["Predictor", "Beta", "Beta_t", "R2", "NumObs"]].to_string(index=False))
    print(f"\nResults saved to: {out_file}")


# =========================================================================
#  MODE 'full': single regression on all predictors (kitchen sink)
# =========================================================================
else:
    keep = ~(np.isnan(y) | np.isnan(X_lag).any(axis=1))
    Xz = zscore_fit(X_lag[keep, :])
    res = sm.OLS(y[keep], sm.add_constant(Xz)).fit()

    print(f"\n========== In-sample kitchen-sink regression ({X_lag.shape[1]} predictors) ==========")
    print(f"  Observations : {int(keep.sum())}")
    print(f"  Intercept    : {res.params[0]:+.4f} (t = {res.tvalues[0]:6.2f})")
    rows = []
    for j, name in enumerate(pred_names):
        b, t = res.params[j + 1], res.tvalues[j + 1]
        rows.append(dict(Predictor=name, Beta=b, Beta_t=t))
        print(f"  {name:22s} {b:+.4f} (t = {t:6.2f})")
    print(f"  R2           : {res.rsquared:.4f}")

    tbl = pd.DataFrame(rows)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(OUTPUT_DIR, f"insample_full_{ts}.csv")
    tbl.to_csv(out_file, index=False)

    summary = {
        "Mode": "full", "TimeLag": TIME_LAG,
        "NumPredictors": X_lag.shape[1], "NumObs": int(keep.sum()),
        "Beg": dates[keep].iloc[0].strftime("%Y-%m-%d"),
        "End": dates[keep].iloc[-1].strftime("%Y-%m-%d"),
        "R2": res.rsquared, "R2_adj": res.rsquared_adj,
        "F_stat": res.fvalue, "F_p": res.f_pvalue, "AIC": res.aic,
    }
    util.save_summary(OUTPUT_DIR, "insample_full", summary)
    print(f"\nCoefficients saved to: {out_file}")
