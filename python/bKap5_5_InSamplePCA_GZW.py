"""
bKap5_5_InSamplePCA_GZW.py
Python port of bKap5_5_InSamplePCA_GZW.m.

PCA dimensionality reduction of the Liedtke macro panel (in-sample). The macro
predictors are reduced to a few principal components and the inflation target is
regressed on those components (PCR).

Run from the repo root:  python python/bKap5_5_InSamplePCA_GZW.py
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
iNumComp    = 3     # number of principal components to retain
iTransformX = 2     # 0 = none, 1 = centre, 2 = z-standardise
SHOW_PLOT   = False  # set True to draw the explained-variance bar chart

# =========================================================================
#  Load data
# =========================================================================
tData = pd.read_csv(DATA_PATH)
cAllNames = list(tData.columns)
dtDates = pd.to_datetime(tData.iloc[:, 0])
mX = tData.iloc[:, 2:].to_numpy(float)     # macro predictors
vY = tData.iloc[:, 1].to_numpy(float)      # inflation target
cXnamesM = cAllNames[2:]

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
#  PCA dimensionality reduction
# =========================================================================
# Full PCA to obtain explained variance for all factors
rModelFull = models.est_pca(mXlag, num_comp=iNumPredictors, transform_x=iTransformX)
vExplVarPerFactor = np.diff(np.concatenate([[0.0], rModelFull["expl_var"]])) * 100

if SHOW_PLOT:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure()
    plt.bar(np.arange(1, iNumPredictors + 1), vExplVarPerFactor)
    plt.xlabel("Factor")
    plt.ylabel("Explained Variance (%)")
    plt.title("Explained Variance by Factor")
    plt.box(False)
    plt.savefig(os.path.join(RESULTS_PATH, "InSamplePCA_ExplVar_py.png"))

# Estimate PCA keeping iNumComp principal components
rModel = models.est_pca(mXlag, num_comp=iNumComp, transform_x=iTransformX)
mScores = rModel["scores"]

# =========================================================================
#  In-sample analysis: regress inflation on the retained PCs (with intercept)
# =========================================================================
reg = models.ols_regress(vY, mScores)
vBeta = reg["beta"]
vBetaT = reg["t"]
dR2 = reg["R2"]
vYhat = reg["yhat"]

# =========================================================================
#  Report
# =========================================================================
print(f"In-sample PCA regression: inflation ~ {iNumComp} principal components")
print(f"  Variance explained by {iNumComp} PCs : {rModel['expl_var'][-1] * 100:6.2f}%")
print(f"  Intercept     : {vBeta[0]:8.4f} (t = {vBetaT[0]:6.2f})")
for k in range(iNumComp):
    print(f"  PC{k + 1:<2d} coef     : {vBeta[k + 1]:8.4f} (t = {vBetaT[k + 1]:6.2f})")
print(f"  R2            : {dR2:8.4f}")

# =========================================================================
#  Save results (Python copy; does not overwrite the MATLAB .mat)
# =========================================================================
os.makedirs(RESULTS_PATH, exist_ok=True)
sio.savemat(os.path.join(RESULTS_PATH, "InSamplePCAResults_py.mat"), {
    "vBeta": vBeta.reshape(-1, 1),
    "vBetaT": vBetaT.reshape(-1, 1),
    "dR2": dR2,
    "vYhat": vYhat.reshape(-1, 1),
    "rModel": {
        "vMeanX": rModel["mean_x"], "vStdX": rModel["std_x"],
        "iNumComp": rModel["num_comp"], "mEigVec": rModel["eigvec"],
        "vEigVal": rModel["eigval"].reshape(-1, 1), "mScores": rModel["scores"],
        "vExplVar": rModel["expl_var"].reshape(-1, 1),
    },
})
