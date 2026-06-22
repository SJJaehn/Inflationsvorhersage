# Python port of the inflation-forecasting scripts

Python equivalents of the MATLAB `inflation_prediction_*.m` scripts (the
bKap3/4/5 files adapted to the Liedtke data). The numerical machinery is
delegated to standard packages rather than re-implementing the MATLAB Utils by
hand:

- OLS / rolling regressions — `sklearn.linear_model.LinearRegression`
- AR lag selection, Mincer-Zarnowitz regression, AIC/BIC — `statsmodels`
- PCA / PCR — `sklearn.decomposition.PCA`
- PLS — `sklearn.cross_decomposition.PLSRegression`
- standardisation — `sklearn.preprocessing.StandardScaler`
- time-series CV for OOS selection — `sklearn.model_selection.TimeSeriesSplit`

Only the Clark-West and Diebold-Mariano tests are coded explicitly in
`util.py`, since neither package ships them.

**Campbell-Thompson (CT) variants are intentionally dropped** — the target is
inflation, which can be negative, so truncating forecasts at zero is not
meaningful. Everything else is reported: OOS R², Clark-West, Diebold-Mariano,
RMSE, MAE, correlation, hit rate and the Mincer-Zarnowitz regression.

## Files

| Python file | MATLAB original | bKap | What it does |
|---|---|---|---|
| `util.py` | `Utils/*` | — | Shared loaders, metrics, CW/DM tests |
| `inflation_single.py` | `inflation_prediction_single.m` | 3 | Each predictor on its own — in-sample or OOS (`MODE`) |
| `inflation_full.py` | `inflation_prediction_full.m` | 3 | "Kitchen-sink" regression on all predictors — in-sample or OOS (`MODE`) |
| `inflation_stepwise.py` | `inflation_prediction_forward_backward.m` | 3 | Forward-backward selection on full-sample OOS metric |
| `inflation_selection.py` | `inflation_prediction_forward_backward_oos.m` | 3 | Forward-backward selection redone per origin (genuinely OOS, TimeSeriesSplit CV) |
| `inflation_ar.py` | `inflation_prediction_ar.m` | 4 | Autoregressive forecast with reporting lag + AIC lag selection |
| `inflation_var.py` | `bKap4_2_*.m`, `bKap4_3_*.m` | 4 | VARX (AR lags + exogenous predictors), AR vs VARX — in-sample or OOS (`MODE`) |
| `inflation_dimreduction.py` | `bKap5_5_*PCA_GZW.m`, `bKap5_6_*.m` | 5 | PCA (PCR) and PLS, in-sample and OOS |

### Reporting lag (VAR/AR)

In `inflation_var.py` (and `inflation_ar.py`) the reporting lag `r` applies
**only to the target's own AR lags** — the regressors become `y[t-(r+1)] …
y[t-(r+p)]`. The exogenous macro predictors are already reporting-lag aligned
upstream (in `DATA/Liedtke/aggregate.py`), so they enter contemporaneously
(`x[t]`) and are not shifted again.

### Modes

Scripts with both modes (`inflation_single`, `inflation_full`, `inflation_var`,
`inflation_dimreduction`) take a `MODE` setting in their `CONFIG` block:
`"oos"` for rolling/expanding one-step-ahead out-of-sample forecasts (with the
historical-mean benchmark and CW/DM tests), or `"insample"` for a single fit on
the whole sample (coefficients, t-stats, R²).

## Running

Run from the `Codes-2` directory (the scripts use `./DATA/...` relative paths):

```bash
python3 python/inflation_single.py
python3 python/inflation_full.py
python3 python/inflation_stepwise.py
python3 python/inflation_selection.py
python3 python/inflation_ar.py
python3 python/inflation_dimreduction.py     # set METHOD / MODE in the CONFIG block
```

Each script has a `CONFIG` block at the top (input CSV, window type, train
length, lag, metric, etc.). Input CSV layout: column 0 = date, column 1 =
target (inflation), columns 2.. = predictors.

### Output structure

Results are written to a fixed directory tree (no timestamps), so re-running a
configuration overwrites the previous output:

```
RESULTS/<type>/<country>/<mode>/<options>/
    results.csv        # primary metrics / per-predictor table
    predictions.csv    # one-step-ahead forecasts (where applicable)
    chart.png          # diagram (single: R² bar chart; where applicable)
    coefficients.csv / selection_freq.csv / steplog.csv  # extras per script
```

- `type`    — model family: `single`, `full`, `AR`, `VAR`, `PCA`, `PLS`,
  `stepwise`, `selection_cv` (TimeSeriesSplit), `selection_split` (single
  validation split).
- `country` — `US` / `UK`, taken from the input CSV path.
- `mode`    — `insample` or `oos`.
- `options` — a chain of the run options, e.g. `train120_rolling_lag1`,
  `comp3_min120_expanding_lag1`.

### Batch runs

Each varying dimension can be overridden from the environment (defaults fall
back to the literal `CONFIG` value), which makes it easy to sweep
country/mode/window/method without editing the source:

```bash
INF_COUNTRY=UK INF_MODE=oos INF_ROLLING=False python3 python/inflation_full.py
INF_METHOD=PCA INF_MODE=insample python3 python/inflation_dimreduction.py
```

Recognised overrides: `INF_COUNTRY` (US/UK), `INF_MODE` (oos/insample),
`INF_ROLLING` (True/False), `INF_METHOD` (PCA/PLS, dimreduction only).
