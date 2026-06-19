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

iNumLags = 1;

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
sPred       = {'infl'};
iIdxPred    = find(strcmpi(cXnamesM,sPred));

% Get the predictor
mY          = vY;

iNumSeries = size(mY, 2);

%% Data preprocessing
% Determine dimensions
[iNumObs, iNumAssets]       = size(vY);
[iNumObsP, iNumPredictors]  = size(mY);

% Note: The predictor data and the market returns have the same time index.
% Therefore, we need to lag the predictor data so that we have a predictive
% regression
mXlag = [NaN(1,iNumPredictors); mY(1:end-1,:)];

%% Out-of-sample analysis
% Settings
iNumIn = 5000;
lRoll  = false;

% Initialize memory
mYhatVAR1       = NaN(iNumObs, iNumSeries);
mYhatVAR_Opt    = NaN(iNumObs, iNumSeries);

% Initialize some memory for saving the predictions
vYhatTemp   = NaN(iNumObs,1);
vYrollTemp  = NaN(iNumObs,1);

% Loop over time
for iIdxT = iNumIn:iNumObs-1
    % Get index
    if lRoll
        vIdxInSample = (iIdxT-iNumIn+1):iIdxT;
    else
        vIdxInSample = 1:iIdxT;
    end
    vIdxOutOfSample = iIdxT + 1;

    % Get data
    mYin = mY(vIdxInSample,:);

    % Estimate VAR(1) model
    rModelVAR1 = fEstVAR(mYin, [], 'iNumLags', iNumLags, 'lEstAlpha', true);

    % Prediction
    mYhatVAR1(vIdxOutOfSample,:) = fPredictVAR(rModelVAR1, 1, []);

    % Estimate optimal VAR
    rModelOpt = fEstOptVAR(mYin, []);
    
    % Prediction
    vYhatTemp(vIdxOutOfSample,:) = fPredictVAR(rModelOpt, 1, []);
    
    vYrollTemp(iIdxT) = mean(vYin);
end

% Performance evaluation
dMSE_VAR1       = mean( (mY - mYhatVAR1).^2,'all','omitmissing');
dMSE_VAR_Opt    = mean( (mY - mYhatVAR_Opt).^2,'all','omitmissing');

% Print progress
fprintf('RMSE VAR(1) %.4f \n',sqrt(dMSE_VAR1));
fprintf('RMSE VAR-Opt %.4f \n',sqrt(dMSE_VAR_Opt));

% Evaluate quality
[rStatsOOS] = fEvaluatePerformanceOOS(vY, vYrollTemp, vYhatTemp);

% Restore path
path(sOldPath);