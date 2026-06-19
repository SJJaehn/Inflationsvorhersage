"""
bKap5_6_InSample.py
Python port of bKap5_6_InSample.m.

PLS dimensionality reduction of the Liedtke macro panel (in-sample). The macro
predictors are reduced to a few PLS components (which, unlike PCA, are extracted
to explain the inflation target) and inflation is regressed on those components.

Run from the repo root:  python python/bKap5_6_InSample.py
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio

import models

# =========================================================================
#  Settings  (mirror the MATLAB header)
# =========================================================================
DATA_PATH    = "./DATA/Liedtke/US/aggregated.csv"
RESULTS_PATH = "./RESULTS/GWZ/"

iTimeLag    = 1     # additional predictive lag (predictors already lag-aligned)
iNumComp    = 3     # number of PLS components to retain
iTransformX = 2     # 0 = none, 1 = centre, 2 = z-standardise

# =========================================================================
#  Load data
# =========================================================================
tData = pd.read_csv(DATA_PATH)
cAllNames = list(tData.columns)
dtDates = pd.to_datetime(tData.iloc[:, 0])
mX = tData.iloc[:, 2:].to_numpy(float)
vY = tData.iloc[:, 1].to_numpy(float)
cXnamesM = cAllNames[2:]

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
#  PLS dimensionality reduction
# =========================================================================
rModel = models.est_pls(vY, mXlag, num_comp=iNumComp, transform_x=iTransformX)
mScores = rModel["scores"]

# =========================================================================
#  In-sample analysis: regress inflation on the retained PLS comps (with const)
# =========================================================================
reg = models.ols_regress(vY, mScores)
vBeta = reg["beta"]
vBetaT = reg["t"]
dR2 = reg["R2"]
vYhat = reg["yhat"]

# =========================================================================
#  Report
# =========================================================================
print(f"In-sample PLS regression: inflation ~ {iNumComp} PLS components")
print(f"  Variance explained by {iNumComp} comps : "
      f"X {rModel['expl_var'][0, -1] * 100:6.2f}% | "
      f"Y {rModel['expl_var'][1, -1] * 100:6.2f}%")
print(f"  Intercept     : {vBeta[0]:8.4f} (t = {vBetaT[0]:6.2f})")
for k in range(iNumComp):
    print(f"  Comp{k + 1:<2d} coef   : {vBeta[k + 1]:8.4f} (t = {vBetaT[k + 1]:6.2f})")
print(f"  R2            : {dR2:8.4f}")

# =========================================================================
#  Save results (Python copy; does not overwrite the MATLAB .mat)
# =========================================================================
os.makedirs(RESULTS_PATH, exist_ok=True)
sio.savemat(os.path.join(RESULTS_PATH, "InSamplePLSResults_py.mat"), {
    "vBeta": vBeta.reshape(-1, 1),
    "vBetaT": vBetaT.reshape(-1, 1),
    "dR2": dR2,
    "vYhat": vYhat.reshape(-1, 1),
    "rModel": {
        "vMeanX": rModel["mean_x"], "vStdX": rModel["std_x"],
        "vMeanY": rModel["mean_y"], "vStdY": rModel["std_y"],
        "iNumComp": rModel["num_comp"], "mWeightsX": rModel["weights_x"],
        "mScores": rModel["scores"], "mExplVar": rModel["expl_var"],
    },
})
