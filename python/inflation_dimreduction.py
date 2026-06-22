"""
inflation_dimreduction.py  (port of bKap5_5/bKap5_6 -- PCA & PLS)

Dimensionality-reduction forecasts of inflation from the macro panel:
  - METHOD = "PCA": principal-component regression (PCR). Components are the
    leading PCs of the (standardised) predictors -> sklearn.decomposition.PCA.
  - METHOD = "PLS": partial least squares. Components are extracted to explain
    the inflation target -> sklearn.cross_decomposition.PLSRegression.

  - MODE = "insample": fit on the whole sample, report the regression of
    inflation on the retained components.
  - MODE = "oos": rolling/expanding one-step-ahead forecast, re-fitting the
    scaler + decomposition + regression each origin using only past data
    (no look-ahead), benchmarked against the historical mean.

Standardisation of the predictors is done with sklearn StandardScaler, fit on
the in-sample rows only (matches iTransformX = 2 in the MATLAB scripts).

CSV format: col 0 = date, col 1 = target (inflation), col 2.. = predictors.
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression

import util

# =========================================================================
#  CONFIG
# =========================================================================
COUNTRY    = util.cfg("COUNTRY", "US")            # "US" or "UK"
CSV_PATH   = f"./DATA/Liedtke/{COUNTRY}/aggregated.csv"
OUTPUT_DIR = "./RESULTS/"

METHOD     = util.cfg("METHOD", "PLS")    # "PCA" (PCR) or "PLS"
MODE       = util.cfg("MODE", "oos")      # "insample" or "oos"

TIME_LAG   = 1         # additional predictive lag
N_COMP     = 3         # number of components to retain
STANDARDIZE = True     # z-standardise predictors (iTransformX = 2)

# OOS-only settings
ROLLING    = util.cfg("ROLLING", False)   # False = expanding, True = rolling
MIN_INSAMPLE = 120     # minimum in-sample obs before forecasting (rolling: window length)
# =========================================================================


def make_scores(method, n_comp, X_train, y_train):
    """Fit (scaler ->) decomposition on the training block.
    Returns a callable transform(X) -> scores and the training scores."""
    scaler = StandardScaler() if STANDARDIZE else None
    Xt = scaler.fit_transform(X_train) if scaler else X_train

    if method == "PCA":
        model = PCA(n_components=n_comp).fit(Xt)
        scores_train = model.transform(Xt)
        expl = model.explained_variance_ratio_[:n_comp].sum()
    elif method == "PLS":
        model = PLSRegression(n_components=n_comp, scale=False).fit(Xt, y_train)
        scores_train = model.transform(Xt)
        # share of X variance captured by the retained PLS components
        expl = np.var(model.x_scores_, axis=0).sum() / np.var(Xt, axis=0).sum()
    else:
        raise ValueError(f"Unknown METHOD: {method}")

    def transform(X):
        Xx = scaler.transform(X) if scaler else X
        return model.transform(Xx)

    return transform, scores_train, expl


def load_clean():
    dates, y, X, names = util.load_data(CSV_PATH)
    X = util.apply_lag(X, TIME_LAG)
    ok = ~(np.isnan(y) | np.isnan(X).any(axis=1))
    return dates[ok].reset_index(drop=True), y[ok], X[ok, :], names


def run_insample():
    _, y, X, _ = load_clean()
    n_comp = min(N_COMP, X.shape[1])
    _, scores, expl = make_scores(METHOD, n_comp, X, y)

    res = sm.OLS(y, sm.add_constant(scores)).fit()
    yhat = res.fittedvalues

    print(f"In-sample {METHOD} regression: inflation ~ {n_comp} components "
          f"({len(y)} obs)")
    print(f"  X variance explained by components: {expl * 100:.2f}%")
    print(f"  Intercept : {res.params[0]:8.4f} (t = {res.tvalues[0]:6.2f})")
    for k in range(n_comp):
        print(f"  Comp{k + 1:<2} coef: {res.params[k + 1]:8.4f} "
              f"(t = {res.tvalues[k + 1]:6.2f})")
    print(f"  R2        : {res.rsquared:8.4f}")

    fq = util.forecast_quality(y, np.asarray(yhat))
    print(f"  RMSE/MAE  : {fq['RMSE']:.4f} / {fq['MAE']:.4f}   Cor: {fq['Cor']:.4f}")

    options = f"comp{n_comp}_lag{TIME_LAG}"
    out_dir = util.result_dir(OUTPUT_DIR, METHOD, COUNTRY, "insample", options)
    summary = {
        "Method": METHOD, "Mode": "insample", "NumComp": n_comp,
        "TimeLag": TIME_LAG, "NumObs": len(y),
        "VarExplained_X": expl, "R2": res.rsquared,
        "RMSE": fq["RMSE"], "MAE": fq["MAE"], "Cor": fq["Cor"],
    }
    sum_path, _ = util.save_summary(out_dir, f"{METHOD.lower()}_insample", summary)
    print(f"\nSummary saved to: {sum_path}")


def run_oos():
    dates, y, X, _ = load_clean()
    n = len(y)
    n_comp = min(N_COMP, X.shape[1])

    yhat = np.full(n, np.nan)
    yhat_bm = np.full(n, np.nan)

    # Match bKap5: predict row t using rows up to t-1 (no look-ahead). The first
    # forecast origin is row MIN_INSAMPLE-1 (0-based), as in MATLAB iNumIn:iNumObs.
    for t in range(MIN_INSAMPLE - 1, n):
        if ROLLING:
            idx = np.arange(t - MIN_INSAMPLE + 1, t)  # last MIN_INSAMPLE-1 rows
        else:
            idx = np.arange(0, t)                      # all past rows
        Xin, yin = X[idx, :], y[idx]
        xout = X[t, :].reshape(1, -1)

        transform, scores_in, _ = make_scores(METHOD, n_comp, Xin, yin)
        scores_out = transform(xout)

        reg = LinearRegression().fit(scores_in, yin)
        yhat[t] = reg.predict(scores_out)[0]
        yhat_bm[t] = yin.mean()

    oos = util.evaluate_oos(y, yhat_bm, yhat)
    fq = util.forecast_quality(y, yhat)
    valid = ~(np.isnan(y) | np.isnan(yhat))
    beg, end = util.oos_date_range(dates, valid)

    window = "rolling" if ROLLING else "expanding"
    print(f"One-step-ahead OOS inflation forecast ({METHOD})")
    print(f"  Comps = {n_comp} | window = {window} (min in-sample {MIN_INSAMPLE}) "
          f"| forecasts = {int(valid.sum())}\n")
    print(f"  OOS R2     : {oos['R2OOS']:.6f}")
    print(f"  CW stat/p  : {oos['CW']:.4f} / {oos['CWp']:.4f}")
    print(f"  DM stat/p  : {oos['DM']:.4f} / {oos['DMp']:.4f}")
    print(f"  RMSE / MAE : {fq['RMSE']:.6f} / {fq['MAE']:.6f}")
    print(f"  Cor/Hit    : {fq['Cor']:.6f} / {fq['HitRate']:.6f}")
    print(f"  OOS period : {beg} to {end}")

    options = f"comp{n_comp}_min{MIN_INSAMPLE}_{window}_lag{TIME_LAG}"
    out_dir = util.result_dir(OUTPUT_DIR, METHOD, COUNTRY, "oos", options)
    summary = {
        "Method": METHOD, "Mode": "oos", "WindowType": window,
        "NumComp": n_comp, "TimeLag": TIME_LAG, "MinInSample": MIN_INSAMPLE,
        "OOS_Beg": beg, "OOS_End": end, "Num_OOS_Obs": int(valid.sum()),
        "R2_OOS": oos["R2OOS"], "CW_stat": oos["CW"], "CW_p": oos["CWp"],
        "DM_stat": oos["DM"], "DM_p": oos["DMp"],
        "RMSE": fq["RMSE"], "MAE": fq["MAE"], "Cor": fq["Cor"], "HitRate": fq["HitRate"],
        "R2_MZ": fq["MZ_R2"], "F_MZ": fq["MZ_F"], "p_MZ": fq["MZ_p"],
    }
    sum_path, _ = util.save_summary(out_dir, f"{METHOD.lower()}_oos", summary)
    pred_path = os.path.join(out_dir, "predictions.csv")
    pd.DataFrame({"Date": dates, "Actual": y, "Forecast": yhat,
                  "Benchmark": yhat_bm}).to_csv(pred_path, index=False)
    print(f"\nSummary saved to:     {sum_path}")
    print(f"Predictions saved to: {pred_path}")


if __name__ == "__main__":
    if MODE == "insample":
        run_insample()
    elif MODE == "oos":
        run_oos()
    else:
        raise ValueError(f"Unknown MODE: {MODE}")
