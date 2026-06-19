"""
util.py
Shared helpers for the inflation-forecasting scripts.

These mirror the MATLAB Utils functions (fEvaluatePerformanceOOS, fClarkWest,
fDieboldMariano, fForecastQuality) closely enough that results are comparable:
  - OOS R2 vs a benchmark, plus Campbell-Thompson adjustment (forecasts
    truncated at 0).
  - Clark-West test (one-sided), Diebold-Mariano test (two-sided, with the
    Harvey-Leybourne-Newbold small-sample correction).
  - RMSE / MAE / correlation / hit-rate and a Mincer-Zarnowitz regression.

Kept deliberately simple/procedural so the scripts can stay script-like.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


# =========================================================================
#  Data loading / preparation
# =========================================================================
def load_data(csv_path):
    """CSV: col 0 = date, col 1 = target, col 2.. = predictors.

    Returns dates (DatetimeIndex), y (1d array), X (2d array, n x k),
    pred_names (list of str).
    """
    df = pd.read_csv(csv_path)
    dates = pd.to_datetime(df.iloc[:, 0])
    y = df.iloc[:, 1].to_numpy(dtype=float)
    X = df.iloc[:, 2:].to_numpy(dtype=float)
    pred_names = list(df.columns[2:])
    return dates, y, X, pred_names


def apply_lag(X, lag):
    """Shift predictors down by `lag` rows so X_lag[t] holds the value from
    t-lag (available at t to predict y[t]). Top `lag` rows become NaN."""
    if lag <= 0:
        return X.copy()
    Xl = np.full_like(X, np.nan, dtype=float)
    Xl[lag:, :] = X[:-lag, :]
    return Xl


def window_index(t, train_obs, rolling):
    """In-sample row indices for an origin whose last in-sample row is `t`."""
    if rolling:
        return np.arange(t - train_obs + 1, t + 1)
    return np.arange(0, t + 1)


# =========================================================================
#  Rolling one-step-ahead OOS forecast for a given predictor block
# =========================================================================
def rolling_oos_forecast(y, X, train_obs, rolling):
    """OLS y ~ const + X, re-estimated each step (sklearn LinearRegression).

    X is n x k (k may be 0 -> benchmark only). A step is used only if the
    whole training window and the next test row have complete data.
    Returns (yhat, yhat_bm), both length n with NaN where not forecast.
    """
    n = len(y)
    yhat = np.full(n, np.nan)
    yhat_bm = np.full(n, np.nan)
    k = 0 if X is None else X.shape[1]

    for t in range(train_obs - 1, n - 1):
        idx = window_index(t, train_obs, rolling)
        i_out = t + 1

        yin = y[idx]
        if np.isnan(yin).any() or np.isnan(y[i_out]):
            continue

        if k == 0:
            yhat[i_out] = yin.mean()
            yhat_bm[i_out] = yin.mean()
            continue

        Xin = X[idx, :]
        xout = X[i_out, :]
        if np.isnan(Xin).any() or np.isnan(xout).any():
            continue

        model = LinearRegression().fit(Xin, yin)
        yhat[i_out] = model.predict(xout.reshape(1, -1))[0]
        yhat_bm[i_out] = yin.mean()

    return yhat, yhat_bm


# =========================================================================
#  Statistical tests
# =========================================================================
def clark_west(y, yhat_bm, yhat_model):
    """Modified Clark-West (2007). Returns (stat, p) one-sided. The benchmark
    is nested in the model, so a positive stat favours the model."""
    m = ~(np.isnan(y) | np.isnan(yhat_bm) | np.isnan(yhat_model))
    y, b, f = y[m], yhat_bm[m], yhat_model[m]
    n = len(y)
    if n < 3:
        return np.nan, np.nan
    d = (y - b) ** 2 - ((y - f) ** 2 - (b - f) ** 2)
    se = d.std(ddof=1) / np.sqrt(n)        # OLS se of the mean (regress on const)
    if se == 0:
        return np.nan, np.nan
    stat = d.mean() / se
    p = 1 - stats.t.cdf(stat, n - 1)       # one-sided
    return stat, p


def diebold_mariano(y, yhat1, yhat2):
    """Diebold-Mariano (1995) with HLN (1997) correction for 1-step forecasts.
    Returns (stat, p) two-sided. d = err1^2 - err2^2 (positive favours model 2)."""
    m = ~(np.isnan(y) | np.isnan(yhat1) | np.isnan(yhat2))
    y, f1, f2 = y[m], yhat1[m], yhat2[m]
    n = len(y)
    if n < 3:
        return np.nan, np.nan
    d = (y - f1) ** 2 - (y - f2) ** 2
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return np.nan, np.nan
    stat = d.mean() / se
    k = np.sqrt((n + 1 - 2 * 1 + (1 * 0) / n) / n)   # HLN, 1-step ahead
    stat = stat * k
    p = 2 * stats.t.cdf(-abs(stat), n - 1)           # two-sided
    return stat, p


# =========================================================================
#  OOS performance bundle (mirrors fEvaluatePerformanceOOS)
# =========================================================================
def evaluate_oos(y, yhat_bm, yhat_model):
    """Returns a dict with R2OOS, R2OOS_CT and the CW / DM tests (plain and
    Campbell-Thompson adjusted). CW is only computed when the matching R2 > 0."""
    m = ~(np.isnan(y) | np.isnan(yhat_bm) | np.isnan(yhat_model))
    yy, bb, ff = y[m], yhat_bm[m], yhat_model[m]

    out = dict(R2OOS=np.nan, R2OOS_CT=np.nan,
               CW=np.nan, CWp=np.nan, CW_CT=np.nan, CWp_CT=np.nan,
               DM=np.nan, DMp=np.nan, DM_CT=np.nan, DMp_CT=np.nan,
               n_oos=int(m.sum()))
    if m.sum() < 3:
        return out

    # Campbell-Thompson: truncate forecasts at zero
    adj_b = np.maximum(bb, 0.0)
    adj_f = np.maximum(ff, 0.0)

    out["R2OOS"] = 1 - np.sum((yy - ff) ** 2) / np.sum((yy - bb) ** 2)
    out["R2OOS_CT"] = 1 - np.sum((yy - adj_f) ** 2) / np.sum((yy - adj_b) ** 2)

    out["DM"], out["DMp"] = diebold_mariano(yy, bb, ff)
    out["DM_CT"], out["DMp_CT"] = diebold_mariano(yy, adj_b, adj_f)

    if out["R2OOS"] > 0:
        out["CW"], out["CWp"] = clark_west(yy, bb, ff)
    if out["R2OOS_CT"] > 0:
        out["CW_CT"], out["CWp_CT"] = clark_west(yy, adj_b, adj_f)

    return out


# =========================================================================
#  Forecast quality (mirrors fForecastQuality)
# =========================================================================
def forecast_quality(y_true, y_est):
    """RMSE / MAE / correlation / hit-rate plus a Mincer-Zarnowitz regression
    of y_true on a constant and y_est (R2, F, p)."""
    m = ~(np.isnan(y_true) | np.isnan(y_est))
    yt, ye = y_true[m], y_est[m]
    out = dict(RMSE=np.nan, MAE=np.nan, Cor=np.nan, HitRate=np.nan,
               MZ_R2=np.nan, MZ_F=np.nan, MZ_p=np.nan)
    if len(yt) < 3:
        return out

    err = yt - ye
    out["RMSE"] = np.sqrt(np.mean(err ** 2))
    out["MAE"] = np.mean(np.abs(err))
    out["HitRate"] = np.mean(np.sign(yt) == np.sign(ye))
    if ye.std() > 1e-12 and yt.std() > 1e-12:
        out["Cor"] = np.corrcoef(yt, ye)[0, 1]

    # Mincer-Zarnowitz: regress y_true on [1, y_est]
    if ye.std() > 1e-6:
        reg = stats.linregress(ye, yt)
        r2 = reg.rvalue ** 2
        nobs = len(yt)
        out["MZ_R2"] = r2
        if r2 < 1:
            out["MZ_F"] = (r2 / (1 - r2)) * (nobs - 2)
        out["MZ_p"] = reg.pvalue          # two-sided slope p == overall F p here
    return out


# =========================================================================
#  Information criteria for an OLS fit on a set of rows
# =========================================================================
def info_criterion(y, X, idx, which="AIC"):
    """In-sample AIC/BIC for OLS (const + X) on rows `idx`. X may be None/empty
    -> intercept-only model. Lower is better."""
    yi = y[idx]
    nobs = len(yi)
    if X is None or X.shape[1] == 0:
        resid = yi - yi.mean()
        kparam = 1
    else:
        Xi = X[idx, :]
        Xd = np.column_stack([np.ones(nobs), Xi])
        beta, *_ = np.linalg.lstsq(Xd, yi, rcond=None)
        resid = yi - Xd @ beta
        kparam = Xd.shape[1]
    sig2 = np.sum(resid ** 2) / nobs
    if sig2 <= 0:
        return -np.inf
    if which == "AIC":
        return nobs * np.log(sig2) + 2 * kparam
    return nobs * np.log(sig2) + kparam * np.log(nobs)   # BIC


# =========================================================================
#  Small IO helpers
# =========================================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_summary(out_dir, prefix, summary_dict):
    """Write a 2-column Key/Value summary CSV and return the path."""
    ensure_dir(out_dir)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{prefix}_summary_{ts}.csv")
    tbl = pd.DataFrame(list(summary_dict.items()), columns=["Key", "Value"])
    tbl.to_csv(path, index=False)
    return path, ts
