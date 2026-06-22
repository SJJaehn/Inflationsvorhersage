"""
util.py
Shared helpers for the inflation-forecasting scripts (Python port of the
MATLAB Utils used in bKap3/4/5).

The heavy lifting is delegated to standard packages instead of re-implementing
the MATLAB utilities by hand:
  - OLS / rolling regressions          -> sklearn.linear_model.LinearRegression
  - Mincer-Zarnowitz regression, AIC/BIC -> statsmodels.api.OLS
  - distributions for the test p-values  -> scipy.stats

Only the Clark-West and Diebold-Mariano tests are coded explicitly, because
neither sklearn nor statsmodels ships a ready-made version.

Note: the Campbell-Thompson (CT, forecast-truncated-at-zero) variants are
intentionally NOT computed here -- the target is inflation, which can be
negative, so truncating at zero is not meaningful. All the other OOS stats
(R2OOS, Clark-West, Diebold-Mariano, RMSE, MAE, correlation, hit rate and the
Mincer-Zarnowitz regression) are reported.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


# =========================================================================
#  Data loading / preparation
# =========================================================================
def load_data(csv_path):
    """CSV: col 0 = date, col 1 = target, col 2.. = predictors.

    Returns dates (Series of Timestamp), y (1d array), X (2d array, n x k),
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
    t-lag (available at t to predict y[t]). Top `lag` rows become NaN.

    Works for 1d or 2d X.
    """
    X = np.asarray(X, dtype=float)
    if lag <= 0:
        return X.copy()
    Xl = np.full_like(X, np.nan)
    Xl[lag:] = X[:-lag]
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

    X is n x k (k may be 0 -> benchmark only). Mirrors the MATLAB scripts
    exactly: iterate over the calendar; a step is used only if the whole window
    AND the next-step row are complete (no NaN), otherwise it is skipped. (When
    the data has no internal gaps this is identical to dropping NaNs first.)
    Returns (yhat, yhat_bm), both length n with NaN where no forecast is made.
    """
    n = len(y)
    yhat = np.full(n, np.nan)
    yhat_bm = np.full(n, np.nan)
    k = 0 if X is None else X.shape[1]

    for t in range(train_obs - 1, n - 1):
        idx = window_index(t, train_obs, rolling)
        i_out = t + 1
        yin = y[idx]
        if np.isnan(yin).any():
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
#  Statistical tests (no off-the-shelf package equivalent)
# =========================================================================
def clark_west(y, yhat_bm, yhat_model):
    """Clark-West (2007) test for nested models. Returns (stat, p) one-sided;
    a positive stat favours the larger model over the historical-mean
    benchmark. The CW adjustment term corrects the MSPE for the noise the extra
    parameters add under the null."""
    m = ~(np.isnan(y) | np.isnan(yhat_bm) | np.isnan(yhat_model))
    y, b, f = y[m], yhat_bm[m], yhat_model[m]
    n = len(y)
    if n < 3:
        return np.nan, np.nan
    f_adj = (y - b) ** 2 - ((y - f) ** 2 - (b - f) ** 2)
    se = f_adj.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return np.nan, np.nan
    stat = f_adj.mean() / se
    p = stats.t.sf(stat, n - 1)            # one-sided
    return stat, p


def diebold_mariano(y, yhat1, yhat2):
    """Diebold-Mariano (1995) for one-step forecasts with the Harvey-Leybourne-
    Newbold (1997) small-sample correction. Returns (stat, p) two-sided;
    d = err1^2 - err2^2, so a positive stat favours model 2."""
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
    h = 1                                   # forecast horizon (1-step)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)  # HLN (1997); = sqrt((n-1)/n)
    stat *= hln
    p = 2 * stats.t.cdf(-abs(stat), n - 1)  # two-sided
    return stat, p


# =========================================================================
#  OOS performance bundle (mirrors fEvaluatePerformanceOOS, without CT)
# =========================================================================
def evaluate_oos(y, yhat_bm, yhat_model):
    """OOS R2 vs the historical-mean benchmark, plus the Clark-West and
    Diebold-Mariano tests. (No Campbell-Thompson variant -- see module docstring.)
    Clark-West is only reported when R2OOS > 0."""
    m = ~(np.isnan(y) | np.isnan(yhat_bm) | np.isnan(yhat_model))
    yy, bb, ff = y[m], yhat_bm[m], yhat_model[m]

    out = dict(R2OOS=np.nan, CW=np.nan, CWp=np.nan,
               DM=np.nan, DMp=np.nan, n_oos=int(m.sum()))
    if m.sum() < 3:
        return out

    out["R2OOS"] = 1 - np.sum((yy - ff) ** 2) / np.sum((yy - bb) ** 2)
    out["DM"], out["DMp"] = diebold_mariano(yy, bb, ff)
    if out["R2OOS"] > 0:
        out["CW"], out["CWp"] = clark_west(yy, bb, ff)
    return out


# =========================================================================
#  Forecast quality (mirrors fForecastQuality)
# =========================================================================
def forecast_quality(y_true, y_est):
    """RMSE / MAE / correlation / hit-rate plus a Mincer-Zarnowitz regression
    of y_true on [const, y_est] (R2, F, p) estimated with statsmodels OLS."""
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

    # Mincer-Zarnowitz: y_true ~ const + y_est
    if ye.std() > 1e-6:
        res = sm.OLS(yt, sm.add_constant(ye)).fit()
        out["MZ_R2"] = res.rsquared
        out["MZ_F"] = res.fvalue
        out["MZ_p"] = res.f_pvalue
    return out


# =========================================================================
#  Information criteria for an OLS fit on a set of rows (statsmodels)
# =========================================================================
def info_criterion(y, X, idx, which="AIC"):
    """In-sample AIC/BIC for OLS (const + X) on rows `idx` via statsmodels.
    X may be None/empty -> intercept-only model. Lower is better."""
    yi = y[idx]
    if X is None or X.shape[1] == 0:
        Xd = np.ones((len(yi), 1))
    else:
        Xd = sm.add_constant(X[idx, :])
    res = sm.OLS(yi, Xd).fit()
    return res.aic if which == "AIC" else res.bic


# =========================================================================
#  Small IO helpers
# =========================================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def cfg(name, default):
    """Read an ``INF_<NAME>`` environment override, falling back to the literal
    default given in a script's CONFIG block. Used to drive batch runs (varying
    country/mode/window/method) without editing the source. The override is cast
    to the type of `default` (bool/int/float/str), so direct runs with no env var
    set behave exactly as the CONFIG literal specifies."""
    v = os.environ.get(f"INF_{name}")
    if v is None:
        return default
    if isinstance(default, bool):
        return v.strip().lower() in ("1", "true", "yes", "y")
    if isinstance(default, int):
        return int(v)
    if isinstance(default, float):
        return float(v)
    return v


def country_from_path(csv_path):
    """Pull the country code ('US'/'UK') out of a .../Liedtke/<country>/... path."""
    for part in os.path.normpath(csv_path).split(os.sep):
        if part.upper() in ("US", "UK"):
            return part.upper()
    return "NA"


def result_dir(root, type_name, country, mode, options):
    """Build (and create) the structured output directory
    ``<root>/<type>/<country>/<mode>/<options>/`` and return it. Files written
    there (results.csv, chart.png, ...) use fixed names, so each new run for the
    same configuration overwrites the previous one."""
    d = os.path.join(root, type_name, country, mode, options)
    ensure_dir(d)
    return d


def window_tag(rolling):
    return "rolling" if rolling else "expanding"


def save_summary(out_dir, prefix, summary_dict):
    """Write the primary results.csv (2-column Key/Value) into out_dir and return
    (path, timestamp). `prefix` is kept for signature compatibility but the file
    is always named results.csv so it is overwritten on re-runs."""
    ensure_dir(out_dir)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, "results.csv")
    pd.DataFrame(list(summary_dict.items()),
                 columns=["Key", "Value"]).to_csv(path, index=False)
    return path, ts


def oos_date_range(dates, valid_mask):
    """First/last OOS date strings ('yyyy-mm-dd') given a boolean valid mask."""
    if not np.any(valid_mask):
        return "", ""
    d = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)[valid_mask]
    return d.iloc[0].strftime("%Y-%m-%d"), d.iloc[-1].strftime("%Y-%m-%d")
