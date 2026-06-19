"""
inflation_ar_insample.py
In-sample AR / VARX estimation of inflation (counterpart to the rolling OOS
inflation_ar.py; mirrors MATLAB bKap4_1 / bKap4_2).

AR mode (VAR_MODE = False):
    y(t) = alpha + sum_k phi_k y(t-(r+k)) + e(t)
    The target on its OWN lags with a reporting (publication) lag r, so the first
    usable lag is y(t-(r+1)). The AR order p can also be selected by AIC.

VARX mode (VAR_MODE = True):
    y(t) = alpha + sum_k phi_k y(t-(r+k)) + theta' x(t) + e(t)
    AR lags PLUS the macro predictors (cols 2..) as exogenous regressors. They are
    already reporting-lag aligned, so they enter contemporaneously. Predictors
    numerically collinear with an AR lag term are dropped automatically.

The fit (coefficients, t-stats, R2, AIC) is done with statsmodels OLS.

CSV format: col 0 = date, col 1 = target, col 2.. = predictors (VARX only).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

# =========================================================================
#  CONFIG — edit here
# =========================================================================
CSV_PATH = "./DATA/Liedtke/US/aggregated.csv"

VAR_MODE = False         # False = AR (target lags only), True = VARX (+ predictors)

REPORT_LAG = 1           # r: first usable lag is y(t-(r+1)) (target only)
LOOKBACK = 1             # p: number of AR lags

OPTIMAL_LOOKBACK = True   # AR only: also report the AIC-optimal order over the grid
LOOKBACK_GRID = range(1, 13)
# =========================================================================

df = pd.read_csv(CSV_PATH)
y = pd.Series(df.iloc[:, 1].to_numpy(dtype=float), name="y")
n = len(y)


def ar_design(p):
    """DataFrame of AR lag columns y(t-(r+1))..y(t-(r+p)) (NaN-padded)."""
    return pd.concat(
        {f"y(t-{REPORT_LAG + 1 + k})": y.shift(REPORT_LAG + 1 + k) for k in range(p)},
        axis=1,
    )


# Exogenous predictors (VARX only); already reporting-lag aligned -> no shift.
Xpred = pd.DataFrame(index=y.index)
if VAR_MODE:
    Xpred = df.iloc[:, 2:].astype(float)

    # Drop predictors numerically collinear with any AR lag term (data-driven,
    # name-agnostic): a predictor identical to y(t-(r+1+k)) makes the design rank
    # deficient. Detect by (near-)perfect correlation over complete rows.
    ar_lags = ar_design(LOOKBACK)
    dropped = []
    for col in list(Xpred.columns):
        rho = ar_lags.corrwith(Xpred[col]).abs().max()
        if rho >= 1 - 1e-8:
            dropped.append(col)
    Xpred = Xpred.drop(columns=dropped)
    if dropped:
        print(f"Dropped {len(dropped)} predictor(s) collinear with the AR lag: "
              f"{', '.join(dropped)}")

print(f"Loaded {n} observations from {CSV_PATH}")
print(f"Model: {'VARX (AR lags + predictors)' if VAR_MODE else 'AR (target lags only)'}")

# =========================================================================
#  In-sample fit (statsmodels OLS on complete cases)
# =========================================================================
X = pd.concat([ar_design(LOOKBACK), Xpred], axis=1)
data = pd.concat([y, X], axis=1).dropna()
model = sm.OLS(data["y"], sm.add_constant(data.drop(columns="y"))).fit()

print(f"\n========== {'VARX' if VAR_MODE else 'AR'}({LOOKBACK}) in-sample, "
      f"reporting lag r={REPORT_LAG} ==========")
print(f"  Observations         : {int(model.nobs)}")
print(f"  Intercept            : {model.params['const']:9.4f} "
      f"(t = {model.tvalues['const']:6.2f})")
for name in X.columns:
    print(f"  {name:22s} : {model.params[name]:9.4f} (t = {model.tvalues[name]:6.2f})")
print(f"  R2                   : {model.rsquared:9.4f}")
print(f"  AIC                  : {model.aic:9.2f}")

# =========================================================================
#  AR-order selection by AIC (AR only, reporting lag held fixed). All orders are
#  scored on the SAME sample (the one valid for the largest order) so the AICs
#  are comparable.
# =========================================================================
if OPTIMAL_LOOKBACK and not VAR_MODE:
    p_max = max(LOOKBACK_GRID)
    full = pd.concat([y, ar_design(p_max)], axis=1).dropna()
    yc = full["y"]
    best_aic, best_p = np.inf, None
    for p in LOOKBACK_GRID:
        cols = [f"y(t-{REPORT_LAG + 1 + k})" for k in range(p)]
        a = sm.OLS(yc, sm.add_constant(full[cols])).fit().aic
        if a < best_aic:
            best_aic, best_p = a, p
    print(f"\nOptimal AR order by AIC over {list(LOOKBACK_GRID)}: p = {best_p}  "
          f"(reporting lag r = {REPORT_LAG}, AIC = {best_aic:.2f}, n = {len(yc)})")
