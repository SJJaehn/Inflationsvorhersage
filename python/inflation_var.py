"""
inflation_var.py  (port of bKap4_2 / bKap4_3 -- VARX)

VARX forecast of inflation: the target is modelled by its OWN autoregressive
lags PLUS the macro predictors as exogenous regressors.

    y[t] = alpha + sum_k phi_k * y[t-(r+k)] + theta' * x[t] + e[t]

IMPORTANT: the reporting lag r (REPORT_LAG) applies ONLY to the target's own AR
terms. The exogenous predictors are already reporting-lag aligned upstream
(DATA/Liedtke/aggregate.py), so they enter contemporaneously (x[t]) and are not
shifted again here.

Two models are reported and compared:
  - AR   : inflation on its own lags only
  - VARX : AR lags + the exogenous macro predictors

Modes:
  - MODE = "insample": one OLS fit on the whole sample (statsmodels OLS),
    reporting coefficients/t-stats, R2 and AIC.
  - MODE = "oos": rolling/expanding one-step-ahead OOS forecast vs the
    historical-mean benchmark, with OOS R2 / CW / DM / RMSE / MAE / Cor / HitRate.

Predictors that are (near-)perfectly collinear with an AR lag term (e.g. a
lagged copy of the target's own series) are dropped, because they make the
VARX design rank deficient.

CSV format: col 0 = date, col 1 = target, col 2.. = predictors.
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

import util

# =========================================================================
#  CONFIG
# =========================================================================
CSV_PATH   = "./DATA/Liedtke/US/aggregated.csv"
OUTPUT_DIR = "./RESULTS/"

MODE       = "oos"     # "insample" or "oos"

REPORT_LAG = 1         # reporting lag r for the AR (target) terms ONLY
NUM_LAGS   = 1         # number of AR lags p

# OOS-only settings
ROLLING      = True   # False = expanding, True = rolling
MIN_INSAMPLE = 240     # minimum in-sample obs before forecasting (rolling: window length)
# =========================================================================


def ar_lag_matrix(y):
    """Columns y[t-(r+1)] .. y[t-(r+p)] (reporting lag on the target only)."""
    lags = [REPORT_LAG + k for k in range(1, NUM_LAGS + 1)]
    return np.column_stack([util.apply_lag(y, L) for L in lags])


def drop_collinear_with_ar(y, X, names):
    """Drop predictors (near-)perfectly correlated with any AR lag term."""
    Ylag = ar_lag_matrix(y)
    keep = np.ones(X.shape[1], dtype=bool)
    for j in range(X.shape[1]):
        for k in range(Ylag.shape[1]):
            ok = ~(np.isnan(X[:, j]) | np.isnan(Ylag[:, k]))
            if ok.sum() > 2:
                a = X[ok, j] - X[ok, j].mean()
                b = Ylag[ok, k] - Ylag[ok, k].mean()
                denom = np.sqrt((a @ a) * (b @ b))
                if denom > 0 and abs(a @ b) / denom >= 1 - 1e-8:
                    keep[j] = False
                    break
    dropped = [names[j] for j in range(len(names)) if not keep[j]]
    if dropped:
        print(f"Dropped {len(dropped)} predictor(s) collinear with the AR lag: "
              + ", ".join(dropped))
    return X[:, keep], [names[j] for j in range(len(names)) if keep[j]]


def fit_ols(y, Xreg, idx):
    """OLS of y on [const, Xreg] over rows `idx` with all values present."""
    ok = idx[~(np.isnan(y[idx]) | np.isnan(Xreg[idx, :]).any(axis=1))]
    return sm.OLS(y[ok], sm.add_constant(Xreg[ok, :])).fit(), ok


def run_insample():
    _, y, X, names = util.load_data(CSV_PATH)
    X = util.apply_lag(X, 1)
    X, names = drop_collinear_with_ar(y, X, names)
    Ylag = ar_lag_matrix(y)

    all_idx = np.arange(len(y))
    res_ar, ok_ar = fit_ols(y, Ylag, all_idx)
    res_vx, ok_vx = fit_ols(y, np.column_stack([Ylag, X]), all_idx)

    print(f"\nAR: inflation ~ AR({NUM_LAGS}, reporting lag {REPORT_LAG})  "
          f"[{len(ok_ar)} obs]")
    print(f"  Intercept      : {res_ar.params[0]:+.4g} (t = {res_ar.tvalues[0]:6.2f})")
    for k in range(NUM_LAGS):
        print(f"  AR y(t-{REPORT_LAG + k + 1:<2})    : "
              f"{res_ar.params[k + 1]:+.4g} (t = {res_ar.tvalues[k + 1]:6.2f})")
    print(f"  R2 / AIC       : {res_ar.rsquared:.4f} / {res_ar.aic:.2f}")

    print(f"\nVARX: inflation ~ AR({NUM_LAGS}, reporting lag {REPORT_LAG}) "
          f"+ {len(names)} exogenous predictors  [{len(ok_vx)} obs]")
    print(f"  Intercept      : {res_vx.params[0]:+.4g} (t = {res_vx.tvalues[0]:6.2f})")
    for k in range(NUM_LAGS):
        print(f"  AR y(t-{REPORT_LAG + k + 1:<2})    : "
              f"{res_vx.params[k + 1]:+.4g} (t = {res_vx.tvalues[k + 1]:6.2f})")
    for j, name in enumerate(names):
        p = res_vx.params[1 + NUM_LAGS + j]
        t = res_vx.tvalues[1 + NUM_LAGS + j]
        print(f"  {name:<32} {p:+.4g} (t = {t:6.2f})")
    print(f"  R2 / AIC       : {res_vx.rsquared:.4f} / {res_vx.aic:.2f}")

    terms = (["Intercept"] + [f"AR_y(t-{REPORT_LAG + k + 1})" for k in range(NUM_LAGS)]
             + names)
    util.ensure_dir(OUTPUT_DIR)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame({"Term": terms, "Coef": res_vx.params,
                  "t_stat": res_vx.tvalues, "p_value": res_vx.pvalues}
                 ).to_csv(os.path.join(OUTPUT_DIR, f"varx_insample_coefs_{ts}.csv"),
                          index=False)
    summary = {
        "Mode": "insample", "ReportLag": REPORT_LAG, "NumLags": NUM_LAGS,
        "NumPredictors": len(names),
        "AR_R2": res_ar.rsquared, "AR_AIC": res_ar.aic, "AR_NumObs": len(ok_ar),
        "VARX_R2": res_vx.rsquared, "VARX_AIC": res_vx.aic, "VARX_NumObs": len(ok_vx),
    }
    sum_path, _ = util.save_summary(OUTPUT_DIR, "varx_insample", summary)
    print(f"\nSummary saved to: {sum_path}")


def run_oos():
    dates, y, X, names = util.load_data(CSV_PATH)
    X = util.apply_lag(X, 1)
    X, names = drop_collinear_with_ar(y, X, names)
    Ylag = ar_lag_matrix(y)
    Xvarx = np.column_stack([Ylag, X])
    n = len(y)

    yhat_ar = np.full(n, np.nan)
    yhat_vx = np.full(n, np.nan)
    yhat_bm = np.full(n, np.nan)

    for t in range(MIN_INSAMPLE - 1, n - 1):
        idx = util.window_index(t, MIN_INSAMPLE, ROLLING)
        i_out = t + 1
        yin = y[idx]
        # Standardized NaN handling (same rule as inflation_ar / inflation_single):
        # use a step only if the whole window AND the next-step row are complete,
        # then OLS on the full window. No complete-row subsetting.
        if np.isnan(yin).any():
            continue
        yhat_bm[i_out] = yin.mean()

        # --- AR model (target lags only) ---
        if not (np.isnan(Ylag[idx, :]).any() or np.isnan(Ylag[i_out, :]).any()):
            res = sm.OLS(yin, sm.add_constant(Ylag[idx, :])).fit()
            yhat_ar[i_out] = float(res.predict(np.r_[1.0, Ylag[i_out, :]])[0])

        # --- VARX model (AR lags + exogenous predictors) ---
        if not (np.isnan(Xvarx[idx, :]).any() or np.isnan(Xvarx[i_out, :]).any()):
            res = sm.OLS(yin, sm.add_constant(Xvarx[idx, :])).fit()
            yhat_vx[i_out] = float(res.predict(np.r_[1.0, Xvarx[i_out, :]])[0])

    window = "rolling" if ROLLING else "expanding"
    print(f"\nOne-step-ahead OOS inflation forecast")
    print(f"  Reporting lag r = {REPORT_LAG} | AR lags p = {NUM_LAGS} | "
          f"window = {window} (min in-sample {MIN_INSAMPLE})\n")

    util.ensure_dir(OUTPUT_DIR)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    summary = {"Mode": "oos", "WindowType": window, "ReportLag": REPORT_LAG,
               "NumLags": NUM_LAGS, "NumPredictors": X.shape[1],
               "MinInSample": MIN_INSAMPLE}

    print(f"  {'Model':<6}{'OOS_R2':>10}{'CW_p':>9}{'DM_p':>9}"
          f"{'RMSE':>11}{'MAE':>11}{'NumObs':>8}")
    for label, fc in (("AR", yhat_ar), ("VARX", yhat_vx)):
        oos = util.evaluate_oos(y, yhat_bm, fc)
        fq = util.forecast_quality(y, fc)
        n_obs = int((~np.isnan(fc)).sum())
        print(f"  {label:<6}{oos['R2OOS']:>10.4f}{oos['CWp']:>9.4f}"
              f"{oos['DMp']:>9.4f}{fq['RMSE']:>11.6f}{fq['MAE']:>11.6f}{n_obs:>8}")
        summary.update({
            f"{label}_R2_OOS": oos["R2OOS"], f"{label}_CW_stat": oos["CW"],
            f"{label}_CW_p": oos["CWp"], f"{label}_DM_stat": oos["DM"],
            f"{label}_DM_p": oos["DMp"], f"{label}_RMSE": fq["RMSE"],
            f"{label}_MAE": fq["MAE"], f"{label}_Cor": fq["Cor"],
            f"{label}_HitRate": fq["HitRate"], f"{label}_NumObs": n_obs,
        })

    pred_path = os.path.join(OUTPUT_DIR, f"varx_oos_predictions_{ts}.csv")
    pd.DataFrame({"Date": dates, "Actual": y, "Forecast_AR": yhat_ar,
                  "Forecast_VARX": yhat_vx, "Benchmark": yhat_bm}
                 ).to_csv(pred_path, index=False)
    sum_path, _ = util.save_summary(OUTPUT_DIR, "varx_oos", summary)
    print(f"\nSummary saved to:     {sum_path}")
    print(f"Predictions saved to: {pred_path}")


def main():
    if MODE == "insample":
        run_insample()
    elif MODE == "oos":
        run_oos()
    else:
        raise ValueError(f"Unknown MODE: {MODE}")


if __name__ == "__main__":
    main()
