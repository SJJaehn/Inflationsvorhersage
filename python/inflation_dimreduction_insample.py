"""
inflation_dimreduction_insample.py
In-sample dimensionality-reduction regressions of inflation (counterpart to the
OOS inflation_dimreduction.py; mirrors MATLAB bKap5_5 / bKap5_6).

Two methods (set MODE below):
  'PCA' : the macro panel is reduced to a few principal components (sklearn PCA,
          unsupervised / target-blind) and inflation is regressed on them (PCR).
  'PLS' : the components are extracted to explain the target (sklearn
          PLSRegression, supervised) and inflation is regressed on them.

The reduction runs on the z-standardised predictors; the regression of inflation
on the retained component scores is fit with statsmodels OLS so coefficient
t-stats and R2 are reported. Variance explained is taken from the sklearn models.

CSV format: col 0 = date, col 1 = target, col 2.. = predictors (already
reporting-lag aligned in DATA/Liedtke/aggregate.py).
"""

import numpy as np
import statsmodels.api as sm
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import util

# =========================================================================
#  CONFIG — edit here
# =========================================================================
CSV_PATH = "./DATA/Liedtke/US/aggregated.csv"

MODE = "PCA"        # 'PCA' (unsupervised PCR) or 'PLS' (supervised)
TIME_LAG = 1        # additional predictive lag (predictors already lag-aligned)
NUM_COMP = 3        # number of components to retain
STANDARDIZE = True  # z-standardise predictors (they live on different scales)
# =========================================================================

if MODE not in ("PCA", "PLS"):
    raise ValueError(f"Unknown MODE: {MODE!r} (use 'PCA' or 'PLS')")

dates, y, X_raw, pred_names = util.load_data(CSV_PATH)
X_lag = util.apply_lag(X_raw, TIME_LAG)

# PCA/PLS need complete rows
keep = ~(np.isnan(y) | np.isnan(X_lag).any(axis=1))
yc = y[keep]
Xc = X_lag[keep, :]
num_comp = min(NUM_COMP, Xc.shape[1])
print(f"Loaded {len(y)} observations, {X_lag.shape[1]} predictors from {CSV_PATH}")
print(f"Complete-case sample: {len(yc)} observations | method: {MODE} | "
      f"components: {num_comp}")

if STANDARDIZE:
    Xc = StandardScaler().fit_transform(Xc)

# =========================================================================
#  Reduction + regression
# =========================================================================
if MODE == "PCA":
    pca = PCA(n_components=num_comp).fit(Xc)
    scores = pca.transform(Xc)
    var_x = pca.explained_variance_ratio_.sum()
    var_y = None
else:
    pls = PLSRegression(n_components=num_comp, scale=False).fit(Xc, yc)
    scores = pls.x_scores_
    # Cumulative X variance captured by the components (PLS centres X internally,
    # so reconstruct the centred X as scores @ loadings') and Y variance captured.
    Xcen = Xc - Xc.mean(0)
    Xrec = pls.x_scores_ @ pls.x_loadings_.T
    var_x = 1 - np.sum((Xcen - Xrec) ** 2) / np.sum(Xcen ** 2)
    var_y = 1 - np.sum((yc - pls.predict(Xc).ravel()) ** 2) / np.sum((yc - yc.mean()) ** 2)

# Regress inflation on the retained scores (statsmodels OLS -> t-stats, R2)
res = sm.OLS(yc, sm.add_constant(scores)).fit()

# =========================================================================
#  Report
# =========================================================================
print(f"\n========== In-sample {MODE} regression: inflation ~ {num_comp} components ==========")
print(f"  Observations  : {int(res.nobs)}")
print(f"  Var explained : X {var_x * 100:6.2f}%"
      + (f" | Y {var_y * 100:6.2f}%" if var_y is not None else ""))
print(f"  Intercept     : {res.params[0]:9.4f} (t = {res.tvalues[0]:6.2f})")
for k in range(num_comp):
    label = f"Comp{k + 1}" if MODE == "PLS" else f"PC{k + 1}"
    print(f"  {label:6s} coef   : {res.params[k + 1]:9.4f} (t = {res.tvalues[k + 1]:6.2f})")
print(f"  R2            : {res.rsquared:9.4f}")
