% Script for out-of-sample one-step-ahead inflation forecasting with the bKap4
% VAR utilities. Two models are compared against the historical-mean benchmark:
%   AR   : inflation on its own lags y(t-(r+1))..y(t-(r+p))   (reporting lag r)
%   VARX : those AR lags PLUS the macro predictors as exogenous regressors
% At each origin the model is re-estimated on a rolling/expanding window and
% used to forecast one step ahead.

% Clear console
clear; clc; close all;

% Set path
sOldPath = path;
addpath('./Utils/');

%% Settings
sDataPath  = './DATA/Liedtke/UK/';
iReportLag = 1;       % reporting/publication lag r for the AR terms
iNumLags   = 11;       % number of AR lags p
iNumIn     = 120;     % minimum in-sample observations before forecasting
lRoll      = true;   % false = expanding window, true = rolling window

%% Load data
% CSV layout: col 1 = observation_date, col 2 = target (inflation),
%             col 3..end = predictors (exogenous).
tData    = readtable([sDataPath, 'aggregated.csv']);
dtDates  = tData{:,1};
if ~isdatetime(dtDates); dtDates = datetime(dtDates); end
vY       = tData{:,2};       % inflation target
mX       = tData{:,3:end};   % exogenous macro predictors

% Predictive lag: shift the exogenous predictors so that row t holds the
% PREVIOUS month's (fully published) values. Standing at the end of month t-1
% we forecast month-t inflation, so using the same-month (end-of-month)
% predictor row would be look-ahead. This matches the Python port
% (util.apply_lag(X,1)). Only the exogenous predictors are shifted; the AR
% terms keep their reporting lag (mYlag below).
iTimeLag = 1;
mX       = [NaN(iTimeLag, size(mX,2)); mX(1:end-iTimeLag,:)];

% Drop exogenous predictors that are collinear with the AR lag terms. The VARX
% includes the reporting-lag regressors y(t-(r+1))..y(t-(r+p)); any predictor
% numerically identical to one of them -- e.g. a lagged copy of the target's own
% series like 'CPI_pct1_lag1' -- makes the design rank deficient. Detect by
% (near-)perfect correlation, independent of naming, so correlated-but-distinct
% transforms (e.g. yearly inflation) are kept.
cXnames  = tData.Properties.VariableNames(3:end);
mYlag    = lagmatrix(vY, (iReportLag+iTimeLag):(iReportLag+iTimeLag+iNumLags-1));
lExcl    = false(1, size(mX,2));
for jX = 1:size(mX,2)
    for kL = 1:size(mYlag,2)
        vOK = ~isnan(mX(:,jX)) & ~isnan(mYlag(:,kL));
        if nnz(vOK) > 2
            vA   = mX(vOK,jX)    - mean(mX(vOK,jX));
            vB   = mYlag(vOK,kL) - mean(mYlag(vOK,kL));
            dRho = (vA'*vB) / sqrt((vA'*vA)*(vB'*vB));
            if abs(dRho) >= 1 - 1e-8; lExcl(jX) = true; break; end
        end
    end
end
cDropped     = cXnames(lExcl);
mX(:, lExcl) = [];
if ~isempty(cDropped)
    fprintf('Dropped %d predictor(s) collinear with the AR lag: %s\n', ...
        numel(cDropped), strjoin(cDropped, ', '));
end

% Pre-filter: keep only rows where y, all AR lags, and all exogenous
% predictors are complete. Matches the bKap3_2 / bKap3_4 approach.
% AR and VARX are then evaluated on the same observations (fair comparison).
mZall    = [mYlag, mX];
lIsNaN   = isnan(vY) | any(isnan(mZall), 2);
vY(lIsNaN)       = [];
mYlag(lIsNaN,:)  = [];
mX(lIsNaN,:)     = [];
dtDates(lIsNaN)  = [];

iNumObs  = numel(vY);
iNumPred = size(mX,2);

%% Out-of-sample analysis
vYhatAR   = NaN(iNumObs,1);    % AR forecast
vYhatVARX = NaN(iNumObs,1);    % VARX forecast
vYhatBM   = NaN(iNumObs,1);    % historical-mean benchmark

for iIdxT = iNumIn : iNumObs-1
    % In-sample window
    if lRoll
        vIdxIn = (iIdxT - iNumIn + 1) : iIdxT;
    else
        vIdxIn = 1 : iIdxT;
    end
    iIdxOut = iIdxT + 1;

    vYin     = vY(vIdxIn);
    mYlagIn  = mYlag(vIdxIn, :);
    vYlagOut = mYlag(iIdxOut, :);

    vYhatBM(iIdxOut) = mean(vYin);

    % --- AR model (target lags only) -----------------------------------
    rAR     = fEstVAR(vYin, mYlagIn, 'iNumLags', 0, 'lEstAlpha', true);
    vYhatAR(iIdxOut) = fPredictVAR(rAR, 1, vYlagOut);

    % --- VARX model (AR lags + exogenous predictors) -------------------
    mZin  = [mYlagIn, mX(vIdxIn, :)];
    vZout = [vYlagOut, mX(iIdxOut, :)];
    rVARX = fEstVAR(vYin, mZin, 'iNumLags', 0, 'lEstAlpha', true);
    vYhatVARX(iIdxOut) = fPredictVAR(rVARX, 1, vZout);
end

%% Performance evaluation
rOOS_AR   = fEvaluatePerformanceOOS(vY, vYhatBM, vYhatAR);
rOOS_VARX = fEvaluatePerformanceOOS(vY, vYhatBM, vYhatVARX);

if lRoll; sWindow = 'rolling'; else; sWindow = 'expanding'; end

fprintf('One-step-ahead OOS inflation forecast\n');
fprintf('  Reporting lag r = %d | AR lags p = %d | window = %s (min in-sample %d)\n\n', ...
    iReportLag, iNumLags, sWindow, iNumIn);
fprintf('  %-6s %10s %10s %9s\n', 'Model', 'OOS_R2', 'OOS_R2_CT', 'NumObs');
fprintf('  %-6s %10.4f %10.4f %9d\n', 'AR',   rOOS_AR.vR2OOS,   rOOS_AR.vR2OOSCT,   sum(~isnan(vYhatAR)));
fprintf('  %-6s %10.4f %10.4f %9d\n', 'VARX', rOOS_VARX.vR2OOS, rOOS_VARX.vR2OOSCT, sum(~isnan(vYhatVARX)));

% Restore path
path(sOldPath);
