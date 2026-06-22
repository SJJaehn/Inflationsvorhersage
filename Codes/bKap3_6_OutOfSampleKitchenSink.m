% Script for performing the out-of-sample regressions in GWZ. This script
% requires all observations for all predictors to be available

% Clear console
clear; clc; close all;

% Clear console
clear; clc; close all;

% Set paths
sOldPath = path;
addpath('./Utils/');
sDataPath = './DATA/Empirical/';
sResultsPath = './RESULTS/GWZ/';

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
mX = mXmonthly;
vY = log(1+CRSPSP_M) - log(1+Rf_M);     % Log market return

% Exclude variables that require a dynamic re-estimation
cExclVars = {'ogap','sntm','fbm','rsvix','shtint'};
mX(:,ismember(cXnamesM,cExclVars))      = [];
cXnamesM(ismember(cXnamesM,cExclVars))  = [];

% Clear variables
clear mXmonthly yyyymm Rf_M CRSPSP_M

%% Data preprocessing
% Determine dimensions
[iNumObs, iNumAssets]       = size(vY);
[iNumObsP, iNumPredictors]  = size(mX);

% Lag data
mXlag       = [NaN(1, iNumPredictors); mX(1:end-1,:)];

%% Settings
iNumIn = 240;                       % Number of in-sample periods (10 years)
iNumOut = 1;                        % Number of forecasting periods 
lRoll = false;                      % Rolling time window

%% Out-of-sample analysis
% Get data
mXtemp      = mXlag;
vYtemp      = vY;
dtDateTemp  = dtDates;

% Remove missing values
lIsNaN = isnan(vYtemp) | any(isnan(mXtemp),2);
vYtemp(lIsNaN)      = [];
mXtemp(lIsNaN,:)    = [];
dtDateTemp(lIsNaN)  = [];

% Number of out-of-sample observations
iNumObsTemp = size(vYtemp,1);

% Initialize memory
vYhatTemp   = NaN(iNumObsTemp,1);
vYrollTemp  = NaN(iNumObsTemp,1);

% Loop over time
for iIdxT = iNumIn:iNumObsTemp
    % Get time indices
    if lRoll
        vIdxInSample = (iIdxT-iNumIn+1):(iIdxT-1);
    else
        vIdxInSample = 1:(iIdxT-1);
    end

    % Get data
    mXin    = mXtemp(vIdxInSample,:);       % t-1
    vXout   = mXtemp(iIdxT,:);              % t-1
    vYin    = vYtemp(vIdxInSample,:);       % t

    % Regress only if all observations are available
    if any(isnan(mXin),'all')
        continue
    end

    % % Filter out correlated predictors
    % [mXin, vIncludeX]   = fFilterMulticolinearity(mXin);
    % vXout               = vXout(vIncludeX);

    % Add constant
    mXin    = [ones(size(mXin,1),1), mXin];
    vXout   = [ones(size(vXout,1),1), vXout];

    % Regression
    vBetaTemp = mXin\vYin;

    % Prediction
    vYhatTemp(iIdxT) = vXout * vBetaTemp;

    % Rolling mean prediction
    vYrollTemp(iIdxT) = mean(vYin);
end   

% Add missing values
mYhat  = [NaN(sum(lIsNaN),1); vYhatTemp];
mYroll = [NaN(sum(lIsNaN),1); vYrollTemp];

% Evaluate quality
[rStatsOOS] = fEvaluatePerformanceOOS(vY,mYroll,mYhat);

% === Create table
% Merge all results and round to two digits
mResults = round([rStatsOOS.vR2OOS * 100; ...
    rStatsOOS.vCWp * 100; ...
    rStatsOOS.vR2OOSCT * 100;...
    rStatsOOS.vCWp_CT * 100],2);

% Make to cell
cTable_OOS = sprintfc('%.2f', mResults');

% Add column header
cTable_OOS = [{'OOS R2', 'CW (p)', 'OOS R2 CT', 'CW (p) CT'}; cTable_OOS];

% === Save results
sFilename = [sResultsPath,'OutOfSampleKitchenSinkResults.mat'];
save(sFilename, "cTable_OOS", 'mYhat','mYroll','rStatsOOS');

% Restore path
path(sOldPath);