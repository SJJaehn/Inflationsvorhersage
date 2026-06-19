"""
bKap4_1_DemoVectorAutoregressiveModel.py
Python port of bKap4_1_DemoVectorAutoregressiveModel.m.

In-sample AR model of inflation with a reporting (publication) lag r, so the
regressors for y(t) are y(t-(r+1)), ..., y(t-(r+p)). The AR order p is also
selected by AIC over a grid.

Run from the repo root:  python python/bKap4_1_DemoVectorAutoregressiveModel.py
"""

import numpy as np
import pandas as pd

import models

# =========================================================================
#  Settings
# =========================================================================
DATA_PATH  = "./DATA/Liedtke/US/aggregated.csv"
iReportLag = 1          # reporting lag r (first usable AR lag = r+1)
iNumLags   = 1          # number of AR lags p (single fit below)
vLagGrid   = range(1, 13)   # candidate AR orders for AIC selection

# =========================================================================
#  Load data
# =========================================================================
tData = pd.read_csv(DATA_PATH)
dtDates = pd.to_datetime(tData.iloc[:, 0])
vY = tData.iloc[:, 1].to_numpy(float)

# =========================================================================
#  Estimation
# =========================================================================
rModelAR = models.est_var(vY, None, num_lags=iNumLags, report_lag=iReportLag,
                          est_alpha=True, get_stats=True)

print(f"AR({iNumLags}) on inflation with reporting lag r = {iReportLag}")
print(f"  Intercept        : {rModelAR['alpha']:8.4f} (t = {rModelAR['alpha_t']:6.2f})")
for k in range(iNumLags):
    print(f"  Coef y(t-{iReportLag + k + 1:<2d})      : "
          f"{rModelAR['phi'][k]:8.4f} (t = {rModelAR['phi_t'][k]:6.2f})")
print(f"  R2               : {rModelAR['R2']:8.4f}")
print(f"  AIC              : {rModelAR['AIC']:8.2f}")

# =========================================================================
#  Select the AR order by AIC (reporting lag held fixed)
# =========================================================================
rModelOpt = models.est_opt_var(vY, None, num_lags_grid=vLagGrid,
                               report_lag=iReportLag)
print(f"\nOptimal AR order by AIC over [{' '.join(map(str, vLagGrid))}]: "
      f"p = {rModelOpt['num_lags']}  (reporting lag r = {iReportLag})")
