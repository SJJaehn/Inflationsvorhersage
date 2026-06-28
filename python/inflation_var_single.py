"""
inflation_var_single.py

VARX forecast of inflation with EACH predictor on its own:
    y[t] = alpha + sum_k phi_k * y[t-(r+k)] + theta * x[t] + e[t]

For every predictor the script runs two models and compares them:
  - AR   : inflation on its own AR lags only (same benchmark for every predictor)
  - VARX : AR lags + that one predictor

OOS mode: rolling/expanding one-step-ahead forecast re-estimated each step.
Results for all predictors are written to one CSV and a bar chart.

CSV format: col 0 = date, col 1 = target, col 2.. = predictors.
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

import util

# =========================================================================
#  CONFIG
# =========================================================================
COUNTRY      = util.cfg("COUNTRY", "US")
CSV_PATH     = f"./DATA/Liedtke/{COUNTRY}/aggregated.csv"
OUTPUT_DIR   = "./RESULTS/"

ROLLING          = util.cfg("ROLLING", True)
MIN_INSAMPLE     = int(util.cfg("MIN_INSAMPLE", 120))
REPORT_LAG       = 1
TIME_LAG         = 1
NUM_LAGS         = int(util.cfg("NUM_LAGS", 1))
SHARED_TIMEFRAME = util.cfg("SHARED_TIMEFRAME", False)  # restrict all predictors to the same complete sample

PLOT_R2      = True
# =========================================================================

_SIG_COLORS  = ["#fdae61", "#f16913", "#b30000"]
_SIG_LABELS  = ["90%", "95%", "99%"]
_SIG_NS      = "#3a6ea5"
_P_THRESH    = [0.10, 0.05, 0.01]
_COLOR_AR    = "#2ca02c"   # green  — AR baseline
_COLOR_FULL  = "#9467bd"   # purple — full VARX (all predictors)
_SPECIAL     = {"AR baseline", "Full VARX (all)"}


def ar_lag_matrix(y):
    lags = [REPORT_LAG + k for k in range(TIME_LAG, TIME_LAG + NUM_LAGS)]
    return np.column_stack([util.apply_lag(y, L) for L in lags])


def is_collinear_with_ar(x, Ylag):
    """True if x is (near-)perfectly correlated with any AR lag column."""
    for k in range(Ylag.shape[1]):
        ok = ~(np.isnan(x) | np.isnan(Ylag[:, k]))
        if ok.sum() > 2:
            a = x[ok] - x[ok].mean()
            b = Ylag[ok, k] - Ylag[ok, k].mean()
            denom = np.sqrt((a @ a) * (b @ b))
            if denom > 0 and abs(a @ b) / denom >= 1 - 1e-8:
                return True
    return False


def run_oos_single(y, Ylag, x, shared_mask=None):
    """OOS loop for AR and VARX(1 predictor). Returns (yhat_ar, yhat_vx, yhat_bm, ok).
    If shared_mask is provided (SHARED_TIMEFRAME), use it instead of per-predictor filtering."""
    Xvarx = np.column_stack([Ylag, x.reshape(-1, 1)])
    ok = shared_mask if shared_mask is not None else (~np.isnan(y) & ~np.isnan(Xvarx).any(axis=1))
    y_f     = y[ok]
    Ylag_f  = Ylag[ok]
    Xvarx_f = Xvarx[ok]
    n = len(y_f)

    yhat_ar = np.full(n, np.nan)
    yhat_vx = np.full(n, np.nan)
    yhat_bm = np.full(n, np.nan)

    for t in range(MIN_INSAMPLE - 1, n - 1):
        idx    = util.window_index(t, MIN_INSAMPLE, ROLLING)
        i_out  = t + 1
        yin    = y_f[idx]

        if ROLLING and (np.isnan(yin).any() or np.isnan(Xvarx_f[idx]).any()):
            continue

        yhat_bm[i_out] = yin.mean()

        res_ar = sm.OLS(yin, sm.add_constant(Ylag_f[idx, :])).fit()
        yhat_ar[i_out] = float(res_ar.predict(np.r_[1.0, Ylag_f[i_out, :]])[0])

        res_vx = sm.OLS(yin, sm.add_constant(Xvarx_f[idx, :])).fit()
        yhat_vx[i_out] = float(res_vx.predict(np.r_[1.0, Xvarx_f[i_out, :]])[0])

    return yhat_ar, yhat_vx, yhat_bm, ok


def run_oos():
    dates, y, X_raw, pred_names = util.load_data(CSV_PATH)
    X_lag = util.apply_lag(X_raw, 1)
    Ylag  = ar_lag_matrix(y)

    if SHARED_TIMEFRAME:
        Xvarx_all  = np.column_stack([Ylag, X_lag])
        shared_mask = ~np.isnan(y) & ~np.isnan(Xvarx_all).any(axis=1)
        print(f"Shared timeframe: {shared_mask.sum()} of {len(y)} observations retained.")
    else:
        shared_mask = None

    rows  = []

    for j, name in enumerate(pred_names):
        x = X_lag[:, j]

        if is_collinear_with_ar(x, Ylag):
            print(f"  [{j+1}/{len(pred_names)}] {name:<30} SKIPPED (collinear with AR lag)")
            continue

        yhat_ar, yhat_vx, yhat_bm, ok_mask = run_oos_single(y, Ylag, x, shared_mask)
        dates_f = dates[ok_mask].reset_index(drop=True)

        oos_ar = util.evaluate_oos(y[ok_mask], yhat_bm, yhat_ar)
        oos_vx = util.evaluate_oos(y[ok_mask], yhat_bm, yhat_vx)
        fq_ar  = util.forecast_quality(y[ok_mask], yhat_ar)
        fq_vx  = util.forecast_quality(y[ok_mask], yhat_vx)

        valid_vx = ~(np.isnan(y[ok_mask]) | np.isnan(yhat_vx))
        beg, end = util.oos_date_range(dates_f, valid_vx)

        rows.append({
            "Predictor":    name,
            "OOS_Beg":      beg, "OOS_End": end,
            "Num_OOS_Obs":  int(valid_vx.sum()),
            # VARX metrics
            "VARX_R2_OOS":  oos_vx["R2OOS"], "VARX_CW_stat": oos_vx["CW"],
            "VARX_CW_p":    oos_vx["CWp"],   "VARX_DM_stat": oos_vx["DM"],
            "VARX_DM_p":    oos_vx["DMp"],   "VARX_RMSE":    fq_vx["RMSE"],
            "VARX_MAE":     fq_vx["MAE"],     "VARX_Cor":     fq_vx["Cor"],
            "VARX_HitRate": fq_vx["HitRate"],
            # AR-only baseline (same sample)
            "AR_R2_OOS":    oos_ar["R2OOS"],  "AR_RMSE":      fq_ar["RMSE"],
            "AR_CW_p":      oos_ar["CWp"],
        })
        print(f"  [{j+1}/{len(pred_names)}] {name:<30} "
              f"VARX R2={oos_vx['R2OOS']:+.4f}  AR R2={oos_ar['R2OOS']:+.4f}  "
              f"RMSE={fq_vx['RMSE']:.4f}")

    out = pd.DataFrame(rows)
    print("\n========== VARX single summary ==========")
    print(out[["Predictor", "VARX_R2_OOS", "VARX_CW_p", "VARX_RMSE", "AR_R2_OOS"]]
          .to_string(index=False))
    return out


def _sig_index_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return -1
    idx = -1
    for i, thr in enumerate(_P_THRESH):
        if p <= thr:
            idx = i
    return idx


def plot_r2_bar(out, out_dir):
    """Bar chart of VARX OOS R² per predictor, sorted descending.
    Coloured by CW significance; table below shows VARX R² and RMSE."""
    col      = "VARX_R2_OOS"
    rmse_col = "VARX_RMSE"
    sig_col  = "VARX_CW_p"
    title    = f"VARX Single-Predictor OOS R²  —  {COUNTRY}"

    keep = [c for c in [col, rmse_col, sig_col, "Predictor"] if c in out.columns]
    d = out[keep].copy()
    d[col]      = d[col]      * 100
    d[rmse_col] = d[rmse_col] * 100

    # Sort: singles by R² descending, AR baseline at its natural position,
    # full VARX pinned last.
    full_mask = d["Predictor"] == "Full VARX (all)"
    rest      = d[~full_mask].sort_values(col, ascending=False)
    d = pd.concat([rest, d[full_mask]]).reset_index(drop=True)

    def _bar_color(name, cw_p):
        if name == "AR baseline":
            return _COLOR_AR
        if name == "Full VARX (all)":
            return _COLOR_FULL
        i = _sig_index_p(cw_p)
        return _SIG_COLORS[i] if i >= 0 else _SIG_NS

    colors = [_bar_color(row["Predictor"], row.get(sig_col))
              for _, row in d.iterrows()]

    n = len(d)
    short_labels = [util.short_name(p) for p in d["Predictor"]]
    max_name_chars = max(len(s) for s in short_labels)
    label_depth_in = max_name_chars * 7.5 * 0.55 / 72
    fig_h = 5.5 + label_depth_in
    fig_w = max(10, 0.6 * n)
    fig = plt.figure(figsize=(fig_w, fig_h))

    tbl_h_frac = 0.28
    tbl_bot    = 0.02
    label_frac = label_depth_in / fig_h
    ax_bot     = tbl_bot + tbl_h_frac + label_frac + 0.03

    ax = fig.add_axes([0.07, ax_bot, 0.91, 0.96 - ax_bot])
    ax.bar(np.arange(n), d[col], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(short_labels, rotation=90, ha="center", fontsize=7.5)
    ax.set_ylabel("OOS R² (in %)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.5, n - 0.5)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=lbl)
               for c, lbl in zip(_SIG_COLORS, _SIG_LABELS)]
    handles.append(Patch(facecolor=_SIG_NS,    label="n.s."))
    handles.append(Patch(facecolor=_COLOR_AR,   label="AR baseline"))
    handles.append(Patch(facecolor=_COLOR_FULL, label="Full VARX (all)"))
    ax.legend(handles=handles, title="Signifikanz (CW)",
              fontsize=8, title_fontsize=8, frameon=False)

    ax_tbl = fig.add_axes([0.07, 0.02, 0.91, 0.30])
    ax_tbl.axis("off")
    tbl = ax_tbl.table(
        cellText=[
            [f"{v:.1f}" for v in d[col]],
            [f"{v:.2f}" for v in d[rmse_col]],
        ],
        rowLabels=["R² (%)", "RMSE (%)"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    for (_, __), cell in tbl.get_celld().items():
        cell.set_height(0.40)

    path = os.path.join(out_dir, "chart.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Chart saved to: {path}")


def main():
    out = run_oos()
    options = (f"min{MIN_INSAMPLE}_{util.window_tag(ROLLING)}_lags{NUM_LAGS}_report{REPORT_LAG}"
               + ("_shared" if SHARED_TIMEFRAME else ""))
    out_dir = util.result_dir(OUTPUT_DIR, "VAR_single", COUNTRY, "oos", options)
    path = os.path.join(out_dir, "results.csv")
    out.to_csv(path, index=False)
    print(f"\nResults saved to: {path}")

    if PLOT_R2:
        out_plot = out.copy()
        var_options = options.replace("_shared", "")
        var_path = os.path.join(OUTPUT_DIR, "VAR", COUNTRY, "oos", var_options, "results.csv")

        def _flt(d, key):
            try: return float(d.get(key, ""))
            except (TypeError, ValueError): return float("nan")

        if SHARED_TIMEFRAME:
            # AR baseline from full VAR results (same shared sample)
            if os.path.exists(var_path):
                fd = pd.read_csv(var_path, index_col=0)["Value"].to_dict()
                ar_row = {"Predictor": "AR baseline",
                          "VARX_R2_OOS": _flt(fd, "AR_R2_OOS"),
                          "VARX_RMSE":   _flt(fd, "AR_RMSE"),
                          "VARX_CW_p":   _flt(fd, "AR_CW_p")}
            else:
                print(f"  (full VAR results not found at {var_path} — skipping AR bar)")
                ar_row = None
        else:
            # AR baseline from standalone AR results (full long sample)
            ar_options = f"train{MIN_INSAMPLE}_{util.window_tag(ROLLING)}_report{REPORT_LAG}_p{NUM_LAGS}"
            ar_path = os.path.join(OUTPUT_DIR, "AR", COUNTRY, "oos", ar_options, "results.csv")
            if os.path.exists(ar_path):
                fd = pd.read_csv(ar_path, index_col=0)["Value"].to_dict()
                ar_row = {"Predictor": "AR baseline",
                          "VARX_R2_OOS": _flt(fd, "R2_OOS"),
                          "VARX_RMSE":   _flt(fd, "RMSE"),
                          "VARX_CW_p":   _flt(fd, "CW_p")}
            else:
                print(f"  (AR results not found at {ar_path} — skipping AR bar)")
                ar_row = None

        extra = []
        if ar_row: extra.append(ar_row)
        if os.path.exists(var_path):
            fd = pd.read_csv(var_path, index_col=0)["Value"].to_dict()
            extra.append({"Predictor": "Full VARX (all)",
                          "VARX_R2_OOS": _flt(fd, "VARX_R2_OOS"),
                          "VARX_RMSE":   _flt(fd, "VARX_RMSE"),
                          "VARX_CW_p":   _flt(fd, "VARX_CW_p")})
        if extra:
            out_plot = pd.concat([out_plot, pd.DataFrame(extra)], ignore_index=True)
        plot_r2_bar(out_plot, out_dir)


if __name__ == "__main__":
    main()
