"""
bKap3_5_InSampleKitchenSink.py
Python port of bKap3_5_InSampleKitchenSink.m.

In-sample "kitchen-sink" regression: inflation on ALL (z-standardised)
predictors at once, using the complete-case sample. Reports beta / t-stat per
predictor and the overall R2, plus the in-sample fitted values.

Run from the repo root:  python python/bKap3_5_InSampleKitchenSink.py
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
#  In-sample analysis (complete-case)
# =========================================================================
lIsNaN = np.isnan(vY) | np.isnan(mXlag).any(axis=1)
keep = ~lIsNaN
vYk = vY[keep]
mXk = mXlag[keep, :]

# z-transformation of each predictor column (sample std, ddof=1)
mXz = (mXk - mXk.mean(axis=0)) / mXk.std(axis=0, ddof=1)

reg = models.ols_regress(vYk, mXz)
vBeta = reg["beta"]      # [intercept, b1..bK]
vBetaT = reg["t"]
dR2 = reg["R2"]

# In-sample fitted values, NaN-padded at the top (removed rows are leading NaNs)
mYhat = np.concatenate([np.full(int(lIsNaN.sum()), np.nan), reg["yhat"]])

# =========================================================================
#  Build the result table (cTable) and save
# =========================================================================
mResults = np.round(np.column_stack([vBeta[1:] * 100, vBetaT[1:]]), 2)
tbl = pd.DataFrame(mResults, columns=["Beta", "Beta T"])
tbl.insert(0, "Predictor", cXnamesM)
print("========== In-sample kitchen-sink results ==========")
print(tbl.to_string(index=False))
print(f"\nR2 : {round(dR2 * 100, 2)}")

os.makedirs(RESULTS_PATH, exist_ok=True)
sio.savemat(os.path.join(RESULTS_PATH, "InSampleKitchenSinkResults_py.mat"), {
    "mYhat": mYhat.reshape(-1, 1), "vBeta": vBeta.reshape(-1, 1),
    "vBetaT": vBetaT.reshape(-1, 1), "dR2": dR2,
})
