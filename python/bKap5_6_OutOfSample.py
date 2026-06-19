"""
bKap5_6_OutOfSample.py
Python port of bKap5_6_OutOfSample.m.

PLS dimensionality reduction (out-of-sample). At each origin a PLS + regression
is re-estimated on a rolling/expanding window and used to forecast inflation one
step ahead, compared against the historical-mean benchmark.

Run from the repo root:  python python/bKap5_6_OutOfSample.py
"""

import numpy as np
import pandas as pd

import models
import util

# =========================================================================
#  Settings  (mirror the MATLAB header)
# =========================================================================
DATA_PATH    = "./DATA/Liedtke/US/aggregated.csv"

iTimeLag    = 1      # additional predictive lag
iNumComp    = 3      # number of PLS components to retain
iTransformX = 2      # 0 = none, 1 = centre, 2 = z-standardise
iNumIn      = 240    # minimum in-sample observations before forecasting
lRoll       = False  # False = expanding window, True = rolling window

# =========================================================================
#  Load data
# =========================================================================
tData = pd.read_csv(DATA_PATH)
dtDates = pd.to_datetime(tData.iloc[:, 0])
mX = tData.iloc[:, 2:].to_numpy(float)
vY = tData.iloc[:, 1].to_numpy(float)

iNumObs, iNumPredictors = mX.shape

# Lag the predictors so they are known one period ahead of the target
mXlag = np.full_like(mX, np.nan)
mXlag[iTimeLag:, :] = mX[:-iTimeLag, :]

# Remove missing values row-wise (PLS does not allow missing values)
lIsNaN = np.isnan(vY) | np.isnan(mXlag).any(axis=1)
keep = ~lIsNaN
vY = vY[keep]
mXlag = mXlag[keep, :]
dtDates = dtDates[keep].reset_index(drop=True)

iNumObs = len(vY)
iNumComp = min(iNumComp, iNumPredictors)

# =========================================================================
#  Out-of-sample analysis (see bKap5_5_OutOfSamplePCA for the index mapping)
# =========================================================================
vYhat = np.full(iNumObs, np.nan)     # PLS forecast
vYroll = np.full(iNumObs, np.nan)    # historical-mean benchmark

for t in range(iNumIn - 1, iNumObs):
    if lRoll:
        idx_in = np.arange(t - iNumIn + 1, t)
    else:
        idx_in = np.arange(0, t)

    mXin = mXlag[idx_in, :]
    vXout = mXlag[t, :]
    vYin = vY[idx_in]

    # Estimate PLS on the in-sample data; project the OOS row with the
    # in-sample mean/std and weights (no look-ahead).
    rModel = models.est_pls(vYin, mXin, num_comp=iNumComp, transform_x=iTransformX)
    mScoresIn = rModel["scores"]
    mScoresOut = models.project_data(vXout, rModel["weights_x"],
                                     rModel["mean_x"], rModel["std_x"])

    # Regress inflation on the in-sample PLS scores (with constant)
    Xd = np.column_stack([np.ones(len(vYin)), mScoresIn])
    beta, *_ = np.linalg.lstsq(Xd, vYin, rcond=None)
    vYhat[t] = np.concatenate([[1.0], mScoresOut.ravel()]) @ beta

    vYroll[t] = vYin.mean()

# =========================================================================
#  Performance evaluation
# =========================================================================
oos = util.evaluate_oos(vY, vYroll, vYhat)
sWindow = "rolling" if lRoll else "expanding"
print("One-step-ahead OOS inflation forecast (PLS)")
print(f"  Comps = {iNumComp} | window = {sWindow} (min in-sample {iNumIn}) | "
      f"forecasts = {int(np.sum(~np.isnan(vYhat)))}\n")

# Table (mirrors cTable_OOS): OOS R2 | CW (p) | OOS R2 CT | CW (p) CT, in %
mResults = np.round([oos["R2OOS"] * 100, oos["CWp"] * 100,
                     oos["R2OOS_CT"] * 100, oos["CWp_CT"] * 100], 2)
header = ["OOS R2", "CW (p)", "OOS R2 CT", "CW (p) CT"]
print("  ".join(f"{h:>10s}" for h in header))
print("  ".join(f"{v:>10.2f}" for v in mResults))
