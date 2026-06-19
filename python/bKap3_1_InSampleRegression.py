"""
bKap3_1_InSampleRegression.py
Python port of bKap3_1_InSampleRegression.m.

In-sample predictive regression of inflation on a single (z-standardised)
predictor, with intercept. Reports beta, t-statistics and R2.

Run from the repo root:  python python/bKap3_1_InSampleRegression.py
"""

import numpy as np
import pandas as pd

import models

# =========================================================================
#  Load data
# =========================================================================
DATA_PATH = "./DATA/Liedtke/US/aggregated.csv"
tData = pd.read_csv(DATA_PATH)
cAllNames = list(tData.columns)
dtDates = pd.to_datetime(tData.iloc[:, 0])

mX = tData.iloc[:, 2:].to_numpy(float)     # predictors
vY = tData.iloc[:, 1].to_numpy(float)      # inflation target
cXnamesM = cAllNames[2:]

# Predictor of interest (default: first predictor)
sPred = cXnamesM[0]
iIdxPred = cXnamesM.index(sPred)
vX = mX[:, iIdxPred]

# =========================================================================
#  Data preprocessing
# =========================================================================
iTimeLag = 1
# Lag the predictor so it is known one period ahead of the target
vXlag = np.full_like(vX, np.nan)
vXlag[iTimeLag:] = vX[:-iTimeLag]

# Remove missing values
lIsNaN = np.isnan(vXlag) | np.isnan(vY)
keep = ~lIsNaN
vYk = vY[keep]
vXk = vXlag[keep]
dtDates = dtDates[keep].reset_index(drop=True)

# z-transformation of predictor (sample std, ddof=1)
vXz = (vXk - vXk.mean()) / vXk.std(ddof=1)

# =========================================================================
#  In-sample regression with intercept
# =========================================================================
reg = models.ols_regress(vYk, vXz)
vBeta = reg["beta"]      # [intercept, slope]
vBetaT = reg["t"]
dR2 = reg["R2"]

print(f"In-sample regression: inflation ~ {sPred} (z-standardised)")
print(f"  Intercept : {vBeta[0]:10.6f} (t = {vBetaT[0]:7.3f})")
print(f"  Beta      : {vBeta[1]:10.6f} (t = {vBetaT[1]:7.3f})")
print(f"  R2        : {dR2:10.6f}")
