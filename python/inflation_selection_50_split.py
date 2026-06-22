"""
inflation_selection.py
Forward-backward predictor selection, redone GENUINELY OUT OF SAMPLE at every
rolling-window origin (no future information leaks into the choice).

At each origin the candidate sets are scored on the in-sample window:
  - holdout metrics (RMSE/R2OOS/...): chronological split where the model fits
    on the first part of the window and is scored on the validation tail (VAL_FRAC).
    This respects time order, so there is no look-ahead.
  - information criteria (AIC/BIC): fit on the FULL window, score in-sample
    (these penalise complexity directly, so no validation split is used).
The selected set is then re-fit on the full window and used to forecast t+1.

Logs per-origin selected predictors and a frequency table of how often each
predictor was chosen.

CSV format: col 0 = date, col 1 = target, col 2.. = predictors.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import util

# =========================================================================
#  CONFIG — edit here
# =========================================================================
COUNTRY    = util.cfg("COUNTRY", "US")            # "US" or "UK"
CSV_PATH   = f"./DATA/Liedtke/{COUNTRY}/aggregated.csv"
OUTPUT_DIR = "./RESULTS/"

ROLLING    = util.cfg("ROLLING", True)
TRAIN_OBS  = 120           # In-sample window length
TIME_LAG   = 1
VAL_FRAC   = 0.5          # Fraction of the window used for validation

# 'R2OOS','RMSE','MAE','Cor','HitRate'  -> scored on validation tail
# 'AIC','BIC'                            -> scored in-sample on the full window
METRIC = "RMSE"
# =========================================================================

HIGHER_BETTER = METRIC in ("R2OOS", "R2OOS_CT", "Cor", "HitRate")
INFO_CRIT = METRIC in ("AIC", "BIC")


# ---- helpers (kept local; they only matter for this script) -------------
def predict_set(y, X, cols, fit_idx, pred_idx):
    """Fit OLS (const + selected cols) on fit_idx, predict pred_idx rows."""
    if len(cols) == 0:
        return np.full(len(pred_idx), y[fit_idx].mean())
    model = LinearRegression().fit(X[np.ix_(fit_idx, cols)], y[fit_idx])
    return model.predict(X[np.ix_(pred_idx, cols)])


def val_score(y, X, cols, win):
    """Chronological split: fit on the first part of the window, score on the 
    validation tail (VAL_FRAC). Used for holdout metrics."""
    n_val = max(1, int(round(len(win) * VAL_FRAC)))
    fit_idx = win[:-n_val]
    test_idx = win[-n_val:]

    yhat = predict_set(y, X, cols, fit_idx, test_idx)
    ytrue = y[test_idx]
    ybm = np.full(len(test_idx), y[fit_idx].mean())

    if len(ytrue) < 3:
        return -np.inf if HIGHER_BETTER else np.inf
        
    if METRIC == "R2OOS":
        val = util.evaluate_oos(ytrue, ybm, yhat)["R2OOS"]
    else:
        val = util.forecast_quality(ytrue, yhat)[METRIC]
        
    if np.isnan(val):
        return -np.inf if HIGHER_BETTER else np.inf
    return val


def score_set(y, X, cols, win):
    """Selection score for a candidate set (lower/higher per METRIC)."""
    if INFO_CRIT:
        # fit on the full window, score in-sample by AIC/BIC (no split)
        Xc = X[:, cols] if len(cols) else None
        return util.info_criterion(y, Xc, win, which=METRIC)
    return val_score(y, X, cols, win)


def better(a, b):
    return a > b if HIGHER_BETTER else a < b


def select_fb(y, X, n_pred, win):
    """Forward-backward greedy selection using score_set."""
    sel = []
    cur = score_set(y, X, sel, win)
    changed = True
    while changed:
        changed = False
        # forward
        best_add, add = cur, -1
        for p in [c for c in range(n_pred) if c not in sel]:
            d = score_set(y, X, sorted(sel + [p]), win)
            if better(d, best_add):
                best_add, add = d, p
        if add >= 0:
            sel = sorted(sel + [add]); cur = best_add; changed = True
        # backward
        rem = True
        while rem and sel:
            rem = False
            best_rem, drop = cur, -1
            for j in range(len(sel)):
                trial = sel[:j] + sel[j + 1:]
                d = score_set(y, X, trial, win)
                if better(d, best_rem):
                    best_rem, drop = d, j
            if drop >= 0:
                sel = sel[:drop] + sel[drop + 1:]; cur = best_rem
                changed = True; rem = True
    return sel


# =========================================================================
#  Load + restrict to the common complete sample
# =========================================================================
dates, y_raw, X_raw, pred_names = util.load_data(CSV_PATH)
n_pred = X_raw.shape[1]
X_lag = util.apply_lag(X_raw, TIME_LAG)

complete = ~(np.isnan(y_raw) | np.isnan(X_lag).any(axis=1))
y = y_raw[complete]
X = X_lag[complete, :]
dates = dates[complete].reset_index(drop=True)
n = len(y)

print(f"Selection metric: {METRIC} "
      f"({'higher' if HIGHER_BETTER else 'lower'} is better, "
      f"{'in-sample/full window' if INFO_CRIT else f'single {int(VAL_FRAC*100)}% validation split'})")
print(f"Common complete sample: {n} of {len(y_raw)} observations retained.")

# =========================================================================
#  Rolling loop with per-origin selection
# =========================================================================
yhat = np.full(n, np.nan)
yhat_bm = np.full(n, np.nan)
num_sel = np.full(n, np.nan)
sel_names = [""] * n
sel_count = np.zeros(n_pred, dtype=int)
n_orig = 0

for t in range(TRAIN_OBS - 1, n - 1):
    win = util.window_index(t, TRAIN_OBS, ROLLING)
    i_out = t + 1
    if np.isnan(y[win]).any() or np.isnan(y[i_out]):
        continue

    sel = select_fb(y, X, n_pred, win)

    # re-fit on the full window, forecast t+1
    yhat[i_out] = predict_set(y, X, sel, win, np.array([i_out]))[0]
    yhat_bm[i_out] = y[win].mean()

    num_sel[i_out] = len(sel)
    sel_names[i_out] = "; ".join(pred_names[c] for c in sel) if sel else "(none)"
    for c in sel:
        sel_count[c] += 1
    n_orig += 1

print(f"Produced {n_orig} out-of-sample forecasts.")

# =========================================================================
#  Metrics + selection frequency
# =========================================================================
oos = util.evaluate_oos(y, yhat_bm, yhat)
fq = util.forecast_quality(y, yhat)
valid = ~(np.isnan(y) | np.isnan(yhat))
beg = dates[valid].iloc[0].strftime("%Y-%m-%d") if valid.any() else ""
end = dates[valid].iloc[-1].strftime("%Y-%m-%d") if valid.any() else ""

print("\n========== Final OOS metrics ==========")
print(f"  OOS R2     : {oos['R2OOS']:.6f}")
print(f"  CW  / p    : {oos['CW']:.4f} / {oos['CWp']:.4f}")
print(f"  DM  / p    : {oos['DM']:.4f} / {oos['DMp']:.4f}")
print(f"  RMSE / MAE : {fq['RMSE']:.6f} / {fq['MAE']:.6f}")
print(f"  Avg # selected / origin: {np.nanmean(num_sel):.2f}")

freq = pd.DataFrame({
    "Predictor": pred_names,
    "TimesSelected": sel_count,
    "FracOrigins": sel_count / n_orig if n_orig else 0.0,
}).sort_values("TimesSelected", ascending=False).reset_index(drop=True)

print("\n========== Predictor selection frequency ==========")
print(freq.to_string(index=False))

# =========================================================================
#  Save
# =========================================================================
options = f"train{TRAIN_OBS}_{util.window_tag(ROLLING)}_{METRIC}_val{int(VAL_FRAC*100)}"
out_dir = util.result_dir(OUTPUT_DIR, "selection_split", COUNTRY, "oos", options)

pd.DataFrame({"Date": dates, "Actual": y, "Forecast": yhat, "Benchmark": yhat_bm,
              "NumSelected": num_sel, "SelectedPredictors": sel_names}
             ).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
freq.to_csv(os.path.join(out_dir, "selection_freq.csv"), index=False)

summary = {
    "Metric": METRIC, "WindowType": "rolling" if ROLLING else "expanding",
    "TrainObs": TRAIN_OBS, "ValFrac": VAL_FRAC, "TimeLag": TIME_LAG,
    "Num_OOS_Obs": int(valid.sum()), "Avg_NumSelected": float(np.nanmean(num_sel)),
    "OOS_Beg": beg, "OOS_End": end,
    "R2_OOS": oos["R2OOS"],
    "CW_stat": oos["CW"], "CW_p": oos["CWp"],
    "DM_stat": oos["DM"], "DM_p": oos["DMp"],
    "RMSE": fq["RMSE"], "MAE": fq["MAE"], "Cor": fq["Cor"], "HitRate": fq["HitRate"],
    "R2_MZ": fq["MZ_R2"], "F_MZ": fq["MZ_F"], "p_MZ": fq["MZ_p"],
}
sum_path, _ = util.save_summary(out_dir, "selection", summary)
print(f"\nResults saved to {out_dir} (results.csv / predictions.csv / selection_freq.csv)")