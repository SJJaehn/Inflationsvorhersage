"""
bKap5_5_OutOfSamplePCA_GZW.py
Python port of bKap5_5_OutOfSamplePCA_GZW.m.

PCA dimensionality reduction (out-of-sample). At each origin a PCA + regression
(PCR) is re-estimated on a rolling/expanding window and used to forecast
inflation one step ahead, compared against the historical-mean benchmark.

Run from the repo root:  python python/bKap5_5_OutOfSamplePCA_GZW.py
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
iNumComp    = 3      # number of principal components to retain
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

# Remove missing values row-wise (PCA does not allow missing values)
lIsNaN = np.isnan(vY) | np.isnan(mXlag).any(axis=1)
keep = ~lIsNaN
vY = vY[keep]
mXlag = mXlag[keep, :]
dtDates = dtDates[keep].reset_index(drop=True)

iNumObs = len(vY)
iNumComp = min(iNumComp, iNumPredictors)

# =========================================================================
#  Out-of-sample analysis
#  MATLAB loops iIdxT = iNumIn:iNumObs (1-based), forecasting the CURRENT iIdxT
#  from the window ending at iIdxT-1. In 0-based terms: t = iNumIn-1 .. iNumObs-1
#  with the in-sample window ending at t-1.
# =========================================================================
vYhat = np.full(iNumObs, np.nan)     # PCR forecast
vYroll = np.full(iNumObs, np.nan)    # historical-mean benchmark

for t in range(iNumIn - 1, iNumObs):
    if lRoll:
        idx_in = np.arange(t - iNumIn + 1, t)     # (iIdxT-iNumIn+1):(iIdxT-1)
    else:
        idx_in = np.arange(0, t)                  # 1:(iIdxT-1)

    mXin = mXlag[idx_in, :]
    vXout = mXlag[t, :]
    vYin = vY[idx_in]

    # Estimate PCA on the in-sample predictors; project the OOS row with the
    # in-sample mean/std and eigenvectors (no look-ahead).
    rModel = models.est_pca(mXin, num_comp=iNumComp, transform_x=iTransformX)
    mScoresIn = rModel["scores"]
    mScoresOut = models.project_data(vXout, rModel["eigvec"],
                                     rModel["mean_x"], rModel["std_x"])

    # PCR: regress inflation on the in-sample scores (with constant)
    Xd = np.column_stack([np.ones(len(vYin)), mScoresIn])
    beta, *_ = np.linalg.lstsq(Xd, vYin, rcond=None)
    vYhat[t] = np.concatenate([[1.0], mScoresOut.ravel()]) @ beta

    vYroll[t] = vYin.mean()

# =========================================================================
#  Performance evaluation
# =========================================================================
oos = util.evaluate_oos(vY, vYroll, vYhat)
sWindow = "rolling" if lRoll else "expanding"
print("One-step-ahead OOS inflation forecast (PCR)")
print(f"  PCs = {iNumComp} | window = {sWindow} (min in-sample {iNumIn}) | "
      f"forecasts = {int(np.sum(~np.isnan(vYhat)))}\n")

# Table (mirrors cTable_OOS): OOS R2 | CW (p) | OOS R2 CT | CW (p) CT, in %
mResults = np.round([oos["R2OOS"] * 100, oos["CWp"] * 100,
                     oos["R2OOS_CT"] * 100, oos["CWp_CT"] * 100], 2)
header = ["OOS R2", "CW (p)", "OOS R2 CT", "CW (p) CT"]
print("  ".join(f"{h:>10s}" for h in header))
print("  ".join(f"{v:>10.2f}" for v in mResults))
