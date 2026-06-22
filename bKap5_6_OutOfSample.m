% Script for applying PLS dimensionality reduction to the Liedtke macro panel
% (out-of-sample). At each origin a PLS + regression is re-estimated on a
% rolling/expanding window and used to forecast inflation one step ahead. The
% forecast is compared against the historical-mean benchmark.

% Clear console
clear; clc; close all;

% Set paths
sOldPath = path;
sCountry = fCfg('COUNTRY', 'US');
sDataPath = ['./DATA/Liedtke/', sCountry, '/'];
sResultsPath = './RESULTS/GWZ/';
addpath('./Utils/');

%% Settings
iTimeLag    = 1;     % additional predictive lag (predictors are already
                     % reporting-lag aligned in DATA/Liedtke/aggregate.py)
iNumComp    = 3;     % number of PLS components to retain
iTransformX = 2;     % preprocessing: 0 = none, 1 = centre, 2 = z-standardise
iNumIn      = 240;   % minimum in-sample observations before forecasting
lRoll       = fCfg('ROLLING', false); % false = expanding window, true = rolling window

%% Load data
% Macro panel produced by DATA/Liedtke/aggregate.py.
% CSV layout: col 1 = observation_date, col 2 = target (inflation),
%             col 3..end = predictors (already lag-aligned in Python).
tData     = readtable([sDataPath, 'aggregated.csv']);
cAllNames = tData.Properties.VariableNames;

% Parse dates (first column)
dtDates = tData{:,1};
if ~isdatetime(dtDates)
    dtDates = datetime(dtDates);
end

% Target (col 2) and predictors (col 3..end)
mX          = tData{:,3:end};          % macro predictors
vY          = tData{:,2};              % inflation target
cXnamesM    = cAllNames(3:end)';       % predictor names as an Nx1 cell

%% Data preprocessing
% Determine dimensions
[iNumObs, iNumAssets]       = size(vY);
[iNumObsP, iNumPredictors]  = size(mX);

% Lag the predictors so that they are known one period ahead of the target
mXlag = [NaN(iTimeLag, iNumPredictors); mX(1:end-iTimeLag,:)];

% Remove missing values row-wise (PLS does not allow missing values)
lIsNaN          = isnan(vY) | any(isnan(mXlag),2);
vY(lIsNaN)      = [];
mXlag(lIsNaN,:) = [];
dtDates(lIsNaN) = [];

% Number of observations
iNumObs  = size(vY,1);
iNumComp = min(iNumComp, iNumPredictors);

%% Out-of-sample analysis
% Initialize memory
vYhat   = NaN(iNumObs,1);     % PLS forecast
vYroll  = NaN(iNumObs,1);     % historical-mean benchmark

% Loop over time
for iIdxT = iNumIn:iNumObs
    % Get time indices
    if lRoll
        vIdxInSample = (iIdxT-iNumIn+1):(iIdxT-1);
    else
        vIdxInSample = 1:(iIdxT-1);
    end

    % Get data
    mXin    = mXlag(vIdxInSample,:);        % t-1
    vXout   = mXlag(iIdxT,:);               % t-1
    vYin    = vY(vIdxInSample,:);           % t

    % Estimate PLS on the in-sample data and project the out-of-sample row
    % using the IN-SAMPLE mean/std and weights (no look-ahead).
    rModel     = fEstPLS(vYin, mXin, "iNumComp", iNumComp, "iTransformX", iTransformX);
    mScoresIn  = rModel.mScores;
    mScoresOut = fProjectData(vXout, rModel.mWeightsX, rModel.vMeanX, rModel.vStdX);

    % Regress inflation on the in-sample PLS scores (with constant)
    mBetaPLS     = [ones(size(mXin,1), 1), mScoresIn] \ vYin;
    vYhat(iIdxT) = [1, mScoresOut] * mBetaPLS;

    % Historical-mean benchmark prediction
    vYroll(iIdxT) = mean(vYin);
end

%% Performance evaluation
[rStatsOOS] = fEvaluatePerformanceOOS(vY, vYroll, vYhat);

if lRoll; sWindow = 'rolling'; else; sWindow = 'expanding'; end
fprintf('One-step-ahead OOS inflation forecast (PLS)\n');
fprintf('  Comps = %d | window = %s (min in-sample %d) | forecasts = %d\n\n', ...
    iNumComp, sWindow, iNumIn, sum(~isnan(vYhat)));

% === Create table
% Merge all results and round to two digits
mResults = round([rStatsOOS.vR2OOS * 100; ...
    rStatsOOS.vCWp * 100; ...
    rStatsOOS.vR2OOSCT * 100; ...
    rStatsOOS.vCWp_CT * 100], 2);

% Make to cell and add column header
cTable_OOS = sprintfc('%.2f', mResults');
cTable_OOS = [{'OOS R2', 'CW (p)', 'OOS R2 CT', 'CW (p) CT'}; cTable_OOS];
disp(cTable_OOS);

% === Save results
% Structured output: <GWZ>/PLS/<country>/oos/<options>/
sCountry = fCountryFromPath(sDataPath);
sOutDir  = fResultDir(sResultsPath, 'PLS', sCountry, 'oos', ...
    sprintf('comp%d_min%d_%s', iNumComp, iNumIn, sWindow));
writecell(cTable_OOS, fullfile(sOutDir, 'results.csv'));
save(fullfile(sOutDir, 'results.mat'), 'cTable_OOS', 'vYhat', 'vYroll', 'rStatsOOS');

% Restore path
path(sOldPath);
