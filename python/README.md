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
length, lag, metric, etc.). Results are written to `./RESULTS/` with a
timestamp. Input CSV layout: column 0 = date, column 1 = target (inflation),
columns 2.. = predictors.
