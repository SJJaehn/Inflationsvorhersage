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
sDataPath  = './DATA/Liedtke/US/';
iReportLag = 1;       % reporting/publication lag r for the AR terms
iNumLags   = 1;       % number of AR lags p
iNumIn     = 240;     % minimum in-sample observations before forecasting
lRoll      = false;   % false = expanding window, true = rolling window

%% Load data
% CSV layout: col 1 = observation_date, col 2 = target (inflation),
%             col 3..end = predictors (exogenous).
tData    = readtable([sDataPath, 'aggregated.csv']);
dtDates  = tData{:,1};
if ~isdatetime(dtDates); dtDates = datetime(dtDates); end
vY       = tData{:,2};       % inflation target
mX       = tData{:,3:end};   % exogenous macro predictors

% Drop exogenous predictors that are collinear with the AR lag terms. The VARX
% includes the reporting-lag regressors y(t-(r+1))..y(t-(r+p)); any predictor
% numerically identical to one of them -- e.g. a lagged copy of the target's own
% series like 'CPI_pct1_lag1' -- makes the design rank deficient. Detect by
% (near-)perfect correlation, independent of naming, so correlated-but-distinct
% transforms (e.g. yearly inflation) are kept.
cXnames  = tData.Properties.VariableNames(3:end);
mYlag    = lagmatrix(vY, (iReportLag+1):(iReportLag+iNumLags));
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

iNumObs  = numel(vY);
iNumPred = size(mX,2);

% A VARX fit needs enough complete (target + all predictors) rows to be
% well-determined; require a comfortable multiple of the parameter count.
iMinCompleteX = 3 * (iNumLags + iNumPred + 1);

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

    vYin = vY(vIdxIn);

    % Benchmark: historical mean of the available target values
    if sum(~isnan(vYin)) < iNumLags + iReportLag + 2
        continue
    end
    vYhatBM(iIdxOut) = mean(vYin, 'omitnan');

    % Recent target tail that the AR lags will read. It must be present and
    % contiguous so fPredictVAR's end-relative indexing stays calendar-correct.
    vTailIdx = (iIdxT - iReportLag - iNumLags + 1) : iIdxT;
    lTailOK  = all(vTailIdx >= 1) && ~any(isnan(vY(vTailIdx)));

    % --- AR model (target lags only) -----------------------------------
    if lTailOK
        rAR     = fEstVAR(vYin, [], 'iNumLags', iNumLags, ...
            'iReportLag', iReportLag, 'lEstAlpha', true);
        mYhatAR = fPredictVAR(rAR, 1, []);
        vYhatAR(iIdxOut) = mYhatAR(1);
    end

    % --- VARX model (AR lags + exogenous predictors) -------------------
    vXout    = mX(iIdxOut, :);
    mXin     = mX(vIdxIn, :);
    lComplX  = ~isnan(vYin) & ~any(isnan(mXin), 2);
    % Need: a complete next-step predictor row, a complete recent tail (so the
    % complete-case target series ends at iIdxT), and enough complete rows.
    if lTailOK && ~any(isnan(vXout)) && ~any(isnan(mX(vTailIdx, :)), 'all') ...
            && sum(lComplX) >= iMinCompleteX
        rVARX     = fEstVAR(vYin, mXin, 'iNumLags', iNumLags, ...
            'iReportLag', iReportLag, 'lEstAlpha', true);
        mYhatVARX = fPredictVAR(rVARX, 1, vXout);
        vYhatVARX(iIdxOut) = mYhatVARX(1);
    end
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
