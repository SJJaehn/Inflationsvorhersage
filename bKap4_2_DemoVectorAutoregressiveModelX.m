% Script for estimating a VARX model of inflation in-sample: the inflation
% target is modelled by its OWN autoregressive lags (using a reporting lag)
% PLUS the macro predictors as EXOGENOUS regressors.
%   y(t) = alpha + sum_k phi_k y(t-(r+k)) + theta' x(t) + e(t)
% The predictors x(t) are already reporting-lag aligned in DATA/Liedtke/aggregate.py.

% Clear console
clear; clc; close all;

% Set path
sOldPath = path;
addpath('./Utils/');

%% Settings
sDataPath  = './DATA/Liedtke/US/';
iReportLag = 1;       % reporting/publication lag r for the AR terms
iNumLags   = 1;       % number of AR lags p

%% Load data
% CSV layout: col 1 = observation_date, col 2 = target (inflation),
%             col 3..end = predictors (exogenous).
tData     = readtable([sDataPath, 'aggregated.csv']);
cAllNames = tData.Properties.VariableNames;
dtDates   = tData{:,1};
if ~isdatetime(dtDates); dtDates = datetime(dtDates); end
vY        = tData{:,2};        % inflation target (endogenous)
mX        = tData{:,3:end};    % macro predictors (exogenous)
cXnames   = cAllNames(3:end);

% Predictive lag: shift the exogenous predictors so that row t holds the
% PREVIOUS month's (fully published) values, matching the Python port
% (util.apply_lag(X,1)). Only the exogenous predictors are shifted; the AR
% terms keep their reporting lag inside fEstVAR.
iTimeLag  = 1;
mX        = [NaN(iTimeLag, size(mX,2)); mX(1:end-iTimeLag,:)];

% Drop exogenous predictors that are collinear with the AR lag terms. The VARX
% includes the reporting-lag regressors y(t-(r+1))..y(t-(r+p)); any predictor
% that is (numerically) identical to one of them -- e.g. a lagged copy of the
% target's own series like 'CPI_pct1_lag1' -- makes the design rank deficient.
% Detect such columns by (near-)perfect correlation, independent of naming, so
% correlated-but-distinct transforms (e.g. yearly inflation) are kept.
mYlag = lagmatrix(vY, (iReportLag+1):(iReportLag+iNumLags));
lExcl = false(1, size(mX,2));
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
cDropped       = cXnames(lExcl);
mX(:, lExcl)   = [];
cXnames(lExcl) = [];
if ~isempty(cDropped)
    fprintf('Dropped %d predictor(s) collinear with the AR lag: %s\n', ...
        numel(cDropped), strjoin(cDropped, ', '));
end

%% Estimation
% Endogenous AR lags y(t-(r+1))..y(t-(r+p)) plus exogenous predictors. fEstVAR
% keeps only rows where the target, its lags, and all predictors are present.
rModelVARX = fEstVAR(vY, mX, 'iNumLags', iNumLags, 'iReportLag', iReportLag, ...
    'lEstAlpha', true, 'lGetStats', true);

fprintf('VARX: inflation ~ AR(%d, reporting lag %d) + %d exogenous predictors\n\n', ...
    iNumLags, iReportLag, size(mX,2));
fprintf('  Intercept        : %8.4f (t = %6.2f)\n', rModelVARX.vAlpha, rModelVARX.vAlphaT);
for k = 1:iNumLags
    fprintf('  AR coef y(t-%-2d)   : %8.4f (t = %6.2f)\n', ...
        iReportLag + k, rModelVARX.mPhi(k), rModelVARX.mPhiT(k));
end
fprintf('  Exogenous predictor coefficients (theta):\n');
for j = 1:numel(cXnames)
    fprintf('    %-34s %8.4f (t = %6.2f)\n', cXnames{j}, ...
        rModelVARX.mTheta(j), rModelVARX.mThetaT(j));
end
fprintf('  R2               : %8.4f\n', rModelVARX.vR2);
fprintf('  AIC              : %8.2f\n', rModelVARX.dAIC);

% Restore path
path(sOldPath);
