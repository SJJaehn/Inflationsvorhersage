"""
bKap3_3_InSampleRegressionAll.py
Python port of bKap3_3_InSampleRegressionAll.m.

In-sample univariate predictive regression of inflation on EACH (z-standardised)
predictor separately. Builds the beta / t-stat / R2 table and the in-sample
fitted values, and saves them (Python copy of InSampleResults.mat).

Run from the repo root:  python python/bKap3_3_InSampleRegressionAll.py
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio

import models

DATA_PATH    = "./DATA/Liedtke/US/aggregated.csv"
RESULTS_PATH = "./RESULTS/GWZ/"

# =========================================================================
#  Load data
# =========================================================================
tData = pd.read_csv(DATA_PATH)
cAllNames = list(tData.columns)
dtDates = pd.to_datetime(tData.iloc[:, 0])
mX = tData.iloc[:, 2:].to_numpy(float)
vY = tData.iloc[:, 1].to_numpy(float)
cXnamesM = cAllNames[2:]

iTimeLag = 1
iNumObs, iNumPredictors = mX.shape

# Lag the predictors so they are known one period ahead of the target
mXlag = np.full_like(mX, np.nan)
mXlag[iTimeLag:, :] = mX[:-iTimeLag, :]

# =========================================================================
#  In-sample analysis, one predictor at a time
# =========================================================================
mYhat = np.full((iNumObs, iNumPredictors), np.nan)
mBeta = np.full((iNumPredictors, 2), np.nan)
mBetaT = np.full((iNumPredictors, 2), np.nan)
vR2 = np.full(iNumPredictors, np.nan)
dtDateSample = [[None, None] for _ in range(iNumPredictors)]

for j in range(iNumPredictors):
    print(f"Estimate in-sample {cXnamesM[j]}")
    vXtemp = mXlag[:, j]
    lIsNaN = np.isnan(vXtemp) | np.isnan(vY)
    keep = ~lIsNaN
    vYk = vY[keep]
    vXk = vXtemp[keep]

    dtDateSample[j] = [dtDates[keep].iloc[0], dtDates[keep].iloc[-1]]

    # z-transformation (sample std, ddof=1)
    vXz = (vXk - vXk.mean()) / vXk.std(ddof=1)

    reg = models.ols_regress(vYk, vXz)
    mBeta[j, :] = reg["beta"]
    mBetaT[j, :] = reg["t"]
    vR2[j] = reg["R2"]
    mYhat[keep, j] = reg["yhat"]

# =========================================================================
#  Build the result table (cTable3) and save
# =========================================================================
mResults = np.round(np.column_stack([mBeta[:, 1] * 100, mBetaT[:, 1], vR2 * 100]), 2)
tbl = pd.DataFrame(mResults, columns=["Beta", "Beta T", "R2"])
tbl.insert(0, "Predictor", cXnamesM)
print("\n========== In-sample results ==========")
print(tbl.to_string(index=False))

os.makedirs(RESULTS_PATH, exist_ok=True)
sio.savemat(os.path.join(RESULTS_PATH, "InSampleResults_py.mat"), {
    "cXnamesM": np.array(cXnamesM, dtype=object).reshape(-1, 1),
    "mBeta": mBeta, "mBetaT": mBetaT, "vR2": vR2.reshape(-1, 1), "mYhat": mYhat,
})
