"""
inflation_stepwise.py  (port of inflation_prediction_forward_backward.m / bKap3)

Forward-backward stepwise predictor selection driven by the OOS metric computed
over the WHOLE sample. At each step it adds the predictor that most improves the
metric, then removes any predictor whose removal improves it further; repeat
until stable. The chosen metric is evaluated with the rolling/expanding OOS loop
(util.rolling_oos_forecast), so selection uses the full-sample OOS performance.

For genuinely out-of-sample (per-origin) selection, use inflation_selection.py.

CSV format: col 0 = date, col 1 = target, col 2.. = predictors.
"""

import os
import numpy as np
import pandas as pd

import util

# =========================================================================
#  CONFIG
# =========================================================================
CSV_PATH   = "./DATA/Liedtke/US/aggregated.csv"
OUTPUT_DIR = "./RESULTS/"

ROLLING   = True
TRAIN_OBS = 240
TIME_LAG  = 1

# 'R2OOS' (higher better) | 'RMSE','MAE' (lower better) | 'Cor','HitRate' (higher)
METRIC = "RMSE"
# =========================================================================

HIGHER_BETTER = METRIC in ("R2OOS", "Cor", "HitRate")


def run_oos(y, X, cols):
    """Full-sample OOS metric for a given predictor-column set."""
    Xsel = X[:, cols] if cols else None
    yhat, yhat_bm = util.rolling_oos_forecast(y, Xsel, TRAIN_OBS, ROLLING)
    if METRIC == "R2OOS":
        val = util.evaluate_oos(y, yhat_bm, yhat)["R2OOS"]
    else:
        val = util.forecast_quality(y, yhat)[METRIC]
    if np.isnan(val):
        return (-np.inf if HIGHER_BETTER else np.inf), yhat, yhat_bm
    return val, yhat, yhat_bm


def better(a, b):
    return a > b if HIGHER_BETTER else a < b


def main():
    dates, y_raw, X_raw, pred_names = util.load_data(CSV_PATH)
    n_pred = X_raw.shape[1]
    X_lag = util.apply_lag(X_raw, TIME_LAG)

    # common complete sample so every candidate set is scored on the same rows
    ok = ~(np.isnan(y_raw) | np.isnan(X_lag).any(axis=1))
    y = y_raw[ok]
    X = X_lag[ok, :]
    dates = dates[ok].reset_index(drop=True)
    print(f"Loaded {len(y_raw)} obs, {n_pred} predictors. "
          f"Common complete sample: {len(y)} obs.")
    print(f"Metric: {METRIC} ({'higher' if HIGHER_BETTER else 'lower'} is better)\n")

    sel = []
    cur, _, _ = run_oos(y, X, sel)
    print(f"Empty model {METRIC} = {cur:.6f}\n")
    step_log = []

    step = 0
    changed = True
    while changed:
        changed = False
        step += 1
        print(f"=== Iteration {step} ===")

        # forward
        best_add, add = cur, -1
        for p in [c for c in range(n_pred) if c not in sel]:
            d, _, _ = run_oos(y, X, sorted(sel + [p]))
            if better(d, best_add):
                best_add, add = d, p
        if add >= 0:
            sel = sorted(sel + [add]); cur = best_add; changed = True
            print(f"  ADD    {pred_names[add]:<24} -> {METRIC} = {cur:.6f}")
            step_log.append((step, "ADD", pred_names[add], cur))
        else:
            print("  No improvement from adding any predictor.")

        # backward
        rem = True
        while rem and sel:
            rem = False
            best_rem, drop = cur, -1
            for j in range(len(sel)):
                d, _, _ = run_oos(y, X, sel[:j] + sel[j + 1:])
                if better(d, best_rem):
                    best_rem, drop = d, j
            if drop >= 0:
                name = pred_names[sel[drop]]
                sel = sel[:drop] + sel[drop + 1:]; cur = best_rem
                changed = rem = True
                print(f"  REMOVE {name:<24} -> {METRIC} = {cur:.6f}")
                step_log.append((step, "REMOVE", name, cur))
        print()

    # final model: all metrics
    print(f"=== Final selected predictors ({len(sel)}) ===")
    print("  " + (", ".join(pred_names[c] for c in sel) if sel else "(none)"))

    _, yhat, yhat_bm = run_oos(y, X, sel)
    oos = util.evaluate_oos(y, yhat_bm, yhat)
    fq = util.forecast_quality(y, yhat)
    valid = ~(np.isnan(y) | np.isnan(yhat))
    beg, end = util.oos_date_range(dates, valid)

    print("\nFinal model metrics:")
    print(f"  OOS R2     : {oos['R2OOS']:.6f}")
    print(f"  CW stat/p  : {oos['CW']:.4f} / {oos['CWp']:.4f}")
    print(f"  DM stat/p  : {oos['DM']:.4f} / {oos['DMp']:.4f}")
    print(f"  RMSE / MAE : {fq['RMSE']:.6f} / {fq['MAE']:.6f}")
    print(f"  Cor/Hit    : {fq['Cor']:.6f} / {fq['HitRate']:.6f}")
    print(f"  OOS period : {beg} to {end}")

    util.ensure_dir(OUTPUT_DIR)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    if step_log:
        pd.DataFrame(step_log, columns=["Step", "Action", "Predictor", "Metric"]
                     ).to_csv(os.path.join(OUTPUT_DIR, f"stepwise_steplog_{ts}.csv"),
                              index=False)

    summary = {
        "Metric": METRIC, "WindowType": "rolling" if ROLLING else "expanding",
        "TrainObs": TRAIN_OBS, "TimeLag": TIME_LAG, "NumSelected": len(sel),
        "SelectedPredictors": "; ".join(pred_names[c] for c in sel),
        "OOS_Beg": beg, "OOS_End": end, "Num_OOS_Obs": int(valid.sum()),
        "R2_OOS": oos["R2OOS"], "CW_stat": oos["CW"], "CW_p": oos["CWp"],
        "DM_stat": oos["DM"], "DM_p": oos["DMp"],
        "RMSE": fq["RMSE"], "MAE": fq["MAE"], "Cor": fq["Cor"], "HitRate": fq["HitRate"],
        "R2_MZ": fq["MZ_R2"], "F_MZ": fq["MZ_F"], "p_MZ": fq["MZ_p"],
    }
    sum_path, _ = util.save_summary(OUTPUT_DIR, "stepwise", summary)
    print(f"\nSummary saved to: {sum_path}")


if __name__ == "__main__":
    main()
