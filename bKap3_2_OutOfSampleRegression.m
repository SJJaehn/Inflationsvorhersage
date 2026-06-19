% Script for performing the out-of-sample regressions in GWZ

% Clear console
clear; clc; close all;

% Set paths
sOldPath = path;
sDataPath = './DATA/Liedtke/US/';
sResultsPath = './RESULTS/GWZ/';
addpath('./Utils/');

%% Settings
iNumIn = 240;                       % Number of in-sample periods (20 years)
iNumOut = 1;                        % Number of forecasting periods
lRoll = true;                      % Rolling time window

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
mX          = tData{:,3:end};          % predictors
vY          = tData{:,2};              % inflation target
cXnamesM    = cAllNames(3:end)';       % predictor names as an Nx1 cell

% Find predictor of interest (edit sPred to any predictor column name)
sPred       = cXnamesM{1};             % default: first predictor
iIdxPred    = find(strcmpi(cXnamesM,sPred));

% Get the predictor
vX          = mX(:,iIdxPred);

%% Data preprocessing
% Predictive lag (periods). The predictors are already reporting-lag aligned
% in Python; this is the additional one-period predictive lag.
iTimeLag = 1;

% Determine dimensions
[iNumObs, iNumAssets]       = size(vY);
[iNumObsP, iNumPredictors]  = size(vX);

% Lag the predictor so that it is known one period ahead of the target
vXlag = [NaN(iTimeLag,iNumPredictors); vX(1:end-iTimeLag,:)];

%% Out-of-Sample analysis
% Remove missing values
lIsNaN          = any(isnan(vXlag),2) | isnan(vY);
vY(lIsNaN,:)    = [];
vXlag(lIsNaN,:) = [];
dtDates(lIsNaN) = [];

% Number of observations
iNumObs         = size(vY,1);

% Initialize some memory for saving the predictions
vYhatTemp   = NaN(iNumObs,1);
vYrollTemp  = NaN(iNumObs,1);

% Loop over time
for iIdxT = iNumIn:iNumObs
    % Get time indices
    if lRoll
        vIdxInSample = (iIdxT-iNumIn+1):(iIdxT-1);
    else
        vIdxInSample = 1:(iIdxT-1);
    end

    % Get data
    vXin    = vXlag(vIdxInSample,:);       % t-1
    vXout   = vXlag(iIdxT,:);              % t-1
    vYin    = vY(vIdxInSample,:);          % t

    % Add constant
    mXin    = [ones(size(vXin,1),1), vXin];
    mXout   = [ones(size(vXout,1),1), vXout];

    % Regression
    vBetaTemp = mXin\vYin;
    % vBetaTemp = regress(vYin, mXin)

    % Prediction
    vYhatTemp(iIdxT) = mXout * vBetaTemp;

    % Rolling mean prediction
    vYrollTemp(iIdxT) = mean(vYin);
end   

% Evaluate quality
[rStatsOOS] = fEvaluatePerformanceOOS(vY, vYrollTemp, vYhatTemp);

% Restore path
path(sOldPath);