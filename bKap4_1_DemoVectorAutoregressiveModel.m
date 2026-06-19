% Script for estimating an autoregressive (AR) model of inflation in-sample.
% The dependent variable is the inflation target. The AR lags use a reporting
% (publication) lag r, so the most recent usable value is y(t-(r+1)):
%   regressors for y(t):  y(t-(r+1)), y(t-(r+2)), ..., y(t-(r+p))
% Built on the bKap4 VAR utilities (fEstVAR / fEstOptVAR), now extended with
% an 'iReportLag' option.

% Clear console
clear; clc; close all;

% Set path
sOldPath = path;
addpath('./Utils/');

%% Settings
sDataPath  = './DATA/Liedtke/US/';
iReportLag = 1;       % reporting/publication lag r (first usable AR lag = r+1)
iNumLags   = 1;       % number of AR lags p (used for the single fit below)
vLagGrid   = 1:12;    % candidate AR orders for AIC-based selection

%% Load data
% CSV layout: col 1 = observation_date, col 2 = target (inflation),
%             col 3..end = predictors (unused in the pure AR model).
tData    = readtable([sDataPath, 'aggregated.csv']);
dtDates  = tData{:,1};
if ~isdatetime(dtDates); dtDates = datetime(dtDates); end
vY       = tData{:,2};       % inflation target

%% Estimation
% AR(p) with reporting lag. fEstVAR removes rows made missing by the lagging
% (and the target's own NaNs) internally.
rModelAR = fEstVAR(vY, [], 'iNumLags', iNumLags, 'iReportLag', iReportLag, ...
    'lEstAlpha', true, 'lGetStats', true);

fprintf('AR(%d) on inflation with reporting lag r = %d\n', iNumLags, iReportLag);
fprintf('  Intercept        : %8.4f (t = %6.2f)\n', rModelAR.vAlpha, rModelAR.vAlphaT);
for k = 1:iNumLags
    fprintf('  Coef y(t-%-2d)      : %8.4f (t = %6.2f)\n', ...
        iReportLag + k, rModelAR.mPhi(k), rModelAR.mPhiT(k));
end
fprintf('  R2               : %8.4f\n', rModelAR.vR2);
fprintf('  AIC              : %8.2f\n', rModelAR.dAIC);

%% Select the AR order by AIC (reporting lag held fixed)
rModelOpt = fEstOptVAR(vY, [], 'vNumLags', vLagGrid, 'iReportLag', iReportLag);
fprintf('\nOptimal AR order by AIC over [%s]: p = %d  (reporting lag r = %d)\n', ...
    num2str(vLagGrid), rModelOpt.iNumLags, iReportLag);

% Restore path
path(sOldPath);
