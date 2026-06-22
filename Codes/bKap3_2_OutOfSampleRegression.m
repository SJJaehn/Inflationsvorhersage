% Script for performing the out-of-sample regressions in GWZ

% Clear console
clear; clc; close all;

% Set paths
sOldPath = path;
sDataPath = './DATA/Empirical/';
sResultsPath = './RESULTS/GWZ/';
addpath('./Utils/');

%% Settings
iNumIn = 240;                       % Number of in-sample periods (20 years)
iNumOut = 1;                        % Number of forecasting periods 
lRoll = false;                      % Rolling time window

%% Load data
% === Predictor data
% File contains monthly, annual, semiannual, and quarterly predictors. We
% use only the monthly predictors
load([sDataPath, 'PredictorDataGWZ.mat'],'mXmonthly','cXnamesM');

% === Market returns
% File contains discrete and log returns. We use only log returns.
load([sDataPath, 'Market.mat'],'CRSPSP_M','yyyymm','Rf_M'); 

% Date is indicated by yyyymm which refers to the END of the respective
% month. However, let us convert this into an actual date format
dtDates = fGetDateFromYYYYMM(yyyymm, 'end');

% Rename data (we use log returns)
mX          = mXmonthly;
vY          = log(1+CRSPSP_M) - log(1+Rf_M);     % Log market return

% Find predictor of interest
sPred       = {'b/m'};
iIdxPred    = find(strcmpi(cXnamesM,sPred));

% Get the predictor
vX          = mX(:,iIdxPred);

%% Data preprocessing
% Determine dimensions
[iNumObs, iNumAssets]       = size(vY);
[iNumObsP, iNumPredictors]  = size(vX);

% Note: The predictor data and the market returns have the same time index.
% Therefore, we need to lag the predictor data so that we have a predictive
% regression
vXlag = [NaN(1,iNumPredictors); vX(1:end-1,:)];

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