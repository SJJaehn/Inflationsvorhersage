"""
models.py
Faithful Python ports of the MATLAB Utils functions used by the bKap3/4/5
scripts that are NOT already covered by util.py:

  - ols_regress          <- regstats2/regstats 'linear' (OLS t-stats, R2)
  - lagmatrix            <- MATLAB lagmatrix
  - est_var / est_opt_var / predict_var   <- fEstVAR / fEstOptVAR / fPredictVAR
  - est_pca / project_data                <- fEstPCA / fProjectData
  - est_pls                               <- fEstPLS (MATLAB plsregress = SIMPLS)

These reproduce the MATLAB maths element-for-element (sample std with ddof=1,
cov with N-1 normalisation, the same design-matrix ordering, the same AIC
formula, SIMPLS weights, ...). The only unavoidable differences are the signs
of eigenvectors / PLS weights, which are arbitrary and cancel out of every
quantity actually reported (fitted values, R2, explained variance); the scores
and coefficients therefore match MATLAB up to a per-component sign.
"""

from typing import Any

import numpy as np


# =========================================================================
#  OLS with a constant (mirrors regstats / regstats2 'linear')
# =========================================================================
def ols_regress(y, X) -> dict[str, Any]:
    """OLS of y on [1, X] (intercept added as the first column).

    Mirrors MATLAB regstats(...,'linear',{'tstat','rsquare'}):
      beta   : (k+1,) coefficients, intercept first
      t      : (k+1,) t-statistics beta/se,  se = sqrt(diag(inv(Xd'Xd))*mse)
      se     : (k+1,) standard errors
      mse    : sse/dfe,  dfe = n-(k+1)
      R2     : 1 - sse/sst   (centred; matches a model with intercept)
      yhat, resid
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = len(y)
    Xd = np.column_stack([np.ones(n), X])
    p = Xd.shape[1]

    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    yhat = Xd @ beta
    resid = y - yhat
    dfe = n - p
    sse = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    mse = sse / dfe
    xtxi = np.linalg.inv(Xd.T @ Xd)
    se = np.sqrt(np.diag(xtxi) * mse)
    t = beta / se
    R2 = 1.0 - sse / sst
    return dict(beta=beta, t=t, se=se, mse=mse, R2=R2, yhat=yhat,
                resid=resid, dfe=dfe)


# =========================================================================
#  MATLAB lagmatrix
# =========================================================================
def lagmatrix(y, lags):
    """Column k = y shifted DOWN by lags[k] (NaN-filled at the top), exactly
    like MATLAB lagmatrix for a single series. `lags` may be empty -> n x 0."""
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    lags = list(lags)
    M = np.full((n, len(lags)), np.nan)
    for k, L in enumerate(lags):
        L = int(L)
        if L >= 0:
            if L < n:
                M[L:, k] = y[: n - L]
        else:  # negative lag = lead
            if -L < n:
                M[: n + L, k] = y[-L:]
    return M


# =========================================================================
#  fEstVAR  (single dependent variable: AR / VARX with exogenous regressors)
# =========================================================================
def est_var(y, X=None, num_lags=1, report_lag=0, est_alpha=True,
            get_stats=False) -> dict[str, Any]:
    """Port of fEstVAR for one dependent variable.

    Regressors: [const?, y(t-(r+1))..y(t-(r+p)), X]. Rows made missing by the
    lagging (and any NaN in y or X) are dropped first, exactly as in MATLAB.

    Returns a dict-like model with: num_lags, report_lag, est_alpha,
    num_dep(=1), num_indep, coef (full beta vector), alpha, alpha_t,
    phi (len p), phi_t, theta (len K), theta_t, R2, AIC, num_prms, y (cleaned).
    """
    y = np.asarray(y, dtype=float).ravel()
    has_x = X is not None and np.size(X) > 0
    if has_x:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        num_indep = X.shape[1]
    else:
        num_indep = 0

    # Lag the dependent variable: y(t-(r+1)) .. y(t-(r+p))
    lags = range(report_lag + 1, report_lag + num_lags + 1)
    Ylag = lagmatrix(y, lags)            # n0 x p (p may be 0)

    # Missing-value mask
    is_nan = np.isnan(Ylag).any(axis=1) | np.isnan(y)
    if has_x:
        is_nan = is_nan | np.isnan(X).any(axis=1)
    keep = ~is_nan

    yk = y[keep]
    Ylagk = Ylag[keep, :]
    n = len(yk)

    # Design matrix [const?, Ylag, X]
    blocks = []
    if est_alpha:
        blocks.append(np.ones((n, 1)))
    if Ylagk.shape[1] > 0:
        blocks.append(Ylagk)
    if has_x:
        blocks.append(X[keep, :])
    Xreg = np.column_stack(blocks) if blocks else np.empty((n, 0))

    beta, *_ = np.linalg.lstsq(Xreg, yk, rcond=None)
    yhat = Xreg @ beta
    resid = yk - yhat

    # t-statistics (same formula as fEstVAR, N=1)
    if get_stats:
        sigma_ols = float(resid @ resid) / (n - Xreg.shape[1])
        xxinv = np.linalg.inv(Xreg.T @ Xreg)
        beta_se = np.sqrt(np.diag(xxinv) * sigma_ols)
        beta_t = beta / beta_se
    else:
        beta_t = np.full_like(beta, np.nan)

    # R2
    tss = float(((yk - yk.mean()) ** 2).sum())
    rss = float(resid @ resid)
    R2 = 1.0 - rss / tss

    # AIC (matches fEstVAR: sigma uses n - num_lags - num_dep)
    sigma = float(resid @ resid) / (n - num_lags - 1)
    num_prms = beta.size
    AIC = 2 * num_prms + (n - num_lags) * np.log(sigma)

    # Decompose beta in the order [const, phi(1..p), theta(1..K)]
    b = beta.copy()
    bt = beta_t.copy()
    idx = 0
    if est_alpha:
        alpha = b[idx]
        alpha_t = bt[idx]
        idx += 1
    else:
        alpha = np.nan
        alpha_t = np.nan
    phi = b[idx:idx + num_lags].copy()
    phi_t = bt[idx:idx + num_lags].copy()
    idx += num_lags
    if has_x:
        theta = b[idx:idx + num_indep].copy()
        theta_t = bt[idx:idx + num_indep].copy()
    else:
        theta = np.array([])
        theta_t = np.array([])

    return dict(num_lags=num_lags, report_lag=report_lag, est_alpha=est_alpha,
                num_dep=1, num_indep=num_indep, coef=beta, alpha=alpha,
                alpha_t=alpha_t, phi=phi, phi_t=phi_t, theta=theta,
                theta_t=theta_t, R2=R2, AIC=AIC, num_prms=num_prms, y=yk)


def est_opt_var(y, X=None, num_lags_grid=range(1, 4), report_lag=0,
                est_alpha=True) -> dict[str, Any]:
    """Port of fEstOptVAR: estimate over a grid of AR orders, return the model
    with the smallest AIC."""
    best = None
    best_aic = np.inf
    for p in num_lags_grid:
        m = est_var(y, X, num_lags=int(p), report_lag=report_lag,
                    est_alpha=est_alpha)
        if m["AIC"] < best_aic:
            best_aic = m["AIC"]
            best = m
    return best


def drop_collinear_with_ar_lags(y, X, names, report_lag, num_lags):
    """Drop exogenous predictors that are (near-)perfectly correlated with one
    of the AR lag terms y(t-(r+1))..y(t-(r+p)) — e.g. a lagged copy of the
    target itself — which would make the VARX design rank deficient. Detection
    is by |corr| >= 1-1e-8 on the overlapping non-NaN rows (data-driven, name
    independent). Returns (X_kept, names_kept, dropped_names)."""
    X = np.asarray(X, dtype=float)
    Ylag = lagmatrix(y, range(report_lag + 1, report_lag + num_lags + 1))
    excl = np.zeros(X.shape[1], dtype=bool)
    for jX in range(X.shape[1]):
        for kL in range(Ylag.shape[1]):
            ok = ~np.isnan(X[:, jX]) & ~np.isnan(Ylag[:, kL])
            if np.count_nonzero(ok) > 2:
                a = X[ok, jX] - X[ok, jX].mean()
                b = Ylag[ok, kL] - Ylag[ok, kL].mean()
                denom = np.sqrt((a @ a) * (b @ b))
                if denom > 0 and abs((a @ b) / denom) >= 1 - 1e-8:
                    excl[jX] = True
                    break
    names = list(names)
    dropped = [names[j] for j in range(len(names)) if excl[j]]
    kept_names = [names[j] for j in range(len(names)) if not excl[j]]
    return X[:, ~excl], kept_names, dropped


def predict_var(model, num_out=1, X=None):
    """Port of fPredictVAR for one dependent variable. Forecasts are appended to
    the working series so multi-step lags feed back correctly."""
    report_lag = model.get("report_lag", 0)
    num_lags = model["num_lags"]
    coef = model["coef"]
    est_alpha = model["est_alpha"]

    series = list(np.asarray(model["y"], dtype=float).ravel())
    iT = len(series)
    if X is not None and np.size(X) > 0:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
    else:
        X = None

    yhat_out = np.full(num_out, np.nan)
    for step in range(1, num_out + 1):
        row = []
        if est_alpha:
            row.append(1.0)
        if num_lags > 0:
            # y(t-(r+1)) .. y(t-(r+p)); end-relative, most-recent first
            for k in range(report_lag + 1, report_lag + num_lags + 1):
                row.append(series[(iT + step) - k - 1])   # -1: 0-based
        if X is not None and X.shape[0] >= step:
            row.extend(X[step - 1, :].tolist())
        xreg = np.asarray(row, dtype=float)
        yh = float(xreg @ coef)
        series.append(yh)
        yhat_out[step - 1] = yh
    return yhat_out


# =========================================================================
#  fProjectData
# =========================================================================
def project_data(X, eigvec, mean_x=None, std_x=None):
    """Standardise X with (mean_x, std_x) then project onto eigvec."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    p = X.shape[1]
    if mean_x is None:
        mean_x = np.zeros(p)
    if std_x is None:
        std_x = np.ones(p)
    Xs = (X - mean_x) / std_x
    return Xs @ eigvec


# =========================================================================
#  fEstPCA
# =========================================================================
def _transform_params(X, transform):
    """Return (mean, std) for transform 0=none, 1=centre, 2=z-standardise."""
    p = X.shape[1]
    if transform == 0:
        return np.zeros(p), np.ones(p)
    if transform == 1:
        return X.mean(axis=0), np.ones(p)
    if transform == 2:
        return X.mean(axis=0), X.std(axis=0, ddof=1)
    raise ValueError("Unknown transformation method")


def est_pca(X, num_comp=1, transform_x=1, frac_var=np.nan,
            positive_mean=False) -> dict[str, Any]:
    """Port of fEstPCA. Eigen-decomposition of the covariance matrix of the
    (optionally standardised) data, components sorted by descending eigenvalue."""
    X = np.asarray(X, dtype=float)
    assert not np.isnan(X).any(), "No missing values allowed"
    mean_x, std_x = _transform_params(X, transform_x)
    Xs = (X - mean_x) / std_x

    sigma = np.cov(Xs, rowvar=False, ddof=1)
    eigval, eigvec = np.linalg.eigh(sigma)          # ascending, symmetric
    order = np.argsort(eigval)[::-1]                 # descending
    eigval = eigval[order]
    eigvec = eigvec[:, order]

    scores = project_data(Xs, eigvec)

    if positive_mean:
        sign = np.sign(scores.mean(axis=0))
        sign[sign == 0] = 1
        scores = scores * sign
        eigvec = eigvec * sign

    expl_var = np.cumsum(eigval) / np.sum(eigval)

    if np.isnan(frac_var):
        nc = int(num_comp)
    else:
        nc = int(np.argmax(expl_var >= frac_var) + 1)

    return dict(num_comp=nc, mean_x=mean_x, std_x=std_x,
                eigvec=eigvec[:, :nc], eigval=eigval[:nc],
                scores=scores[:, :nc], expl_var=expl_var[:nc])


# =========================================================================
#  MATLAB plsregress (SIMPLS) + fEstPLS
# =========================================================================
def _simpls(X0, Y0, ncomp):
    """SIMPLS algorithm, mirroring MATLAB's plsregress subfunction.
    X0, Y0 are already column-centred. Returns (Xloadings, Yloadings, W)."""
    n, dx = X0.shape
    dy = Y0.shape[1]
    Xloadings = np.zeros((dx, ncomp))
    Yloadings = np.zeros((dy, ncomp))
    W = np.zeros((dx, ncomp))
    V = np.zeros((dx, ncomp))

    Cov = X0.T @ Y0
    for i in range(ncomp):
        U, S, Vt = np.linalg.svd(Cov, full_matrices=False)
        ri = U[:, 0]
        ci = Vt[0, :]
        si = S[0]
        ti = X0 @ ri
        normti = np.linalg.norm(ti)
        ti = ti / normti
        Xloadings[:, i] = X0.T @ ti
        qi = si * ci / normti
        Yloadings[:, i] = qi
        W[:, i] = ri / normti

        vi = Xloadings[:, i].copy()
        for _ in range(2):                       # modified Gram-Schmidt (twice)
            for j in range(i):
                vj = V[:, j]
                vi = vi - (vj @ vi) * vj
        vi = vi / np.linalg.norm(vi)
        V[:, i] = vi

        Cov = Cov - np.outer(vi, vi @ Cov)
        Vi = V[:, :i + 1]
        Cov = Cov - Vi @ (Vi.T @ Cov)
    return Xloadings, Yloadings, W


def plsregress(X, Y, ncomp):
    """Subset of MATLAB plsregress: returns (W, pctVar) where W = stats.W are
    the predictor weights and pctVar is the 2 x ncomp explained-variance matrix
    (row 0 = X, row 1 = Y)."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    X0 = X - X.mean(axis=0)
    Y0 = Y - Y.mean(axis=0)
    Xloadings, Yloadings, W = _simpls(X0, Y0, ncomp)
    pctVar = np.vstack([
        (Xloadings ** 2).sum(axis=0) / (X0 ** 2).sum(),
        (Yloadings ** 2).sum(axis=0) / (Y0 ** 2).sum(),
    ])
    return W, pctVar


def est_pls(Y, X, num_comp=1, transform_x=1, transform_y=1, max_comp=np.nan,
            frac_var=np.nan, positive_mean=False) -> dict[str, Any]:
    """Port of fEstPLS (uses SIMPLS, like MATLAB plsregress)."""
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    assert not np.isnan(X).any(), "No missing values allowed"

    n, num_indep = X.shape
    num_dep = Y.shape[1]
    if np.isnan(max_comp):
        max_comp = min(max(num_indep, num_dep), n) - 1
    max_comp = int(max_comp)

    mean_x, std_x = _transform_params(X, transform_x)
    mean_y, std_y = _transform_params(Y, transform_y)
    Xs = (X - mean_x) / std_x
    Ys = (Y - mean_y) / std_y

    W, pctVar = plsregress(Xs, Ys, max_comp)
    scores = project_data(Xs, W)

    if positive_mean:
        sign = np.sign(scores.mean(axis=0))
        sign[sign == 0] = 1
        scores = scores * sign
        W = W * sign

    cum_var = np.cumsum(pctVar, axis=1)

    if np.isnan(frac_var):
        nc = int(num_comp)
    else:
        nc = int(np.argmax(np.all(cum_var >= frac_var, axis=0)) + 1)

    return dict(num_comp=nc, mean_x=mean_x, std_x=std_x, mean_y=mean_y,
                std_y=std_y, weights_x=W[:, :nc], scores=scores[:, :nc],
                expl_var=cum_var[:, :nc])
