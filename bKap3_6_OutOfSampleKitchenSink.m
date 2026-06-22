% Script for performing the out-of-sample regressions in GWZ. This script
% requires all observations for all predictors to be available

% Clear console
clear; clc; close all;

% Clear console
clear; clc; close all;

% Set paths
sOldPath = path;
addpath('./Utils/');
sDataPath = './DATA/Liedtke/US/';
sResultsPath = './RESULTS/GWZ/';

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
mX       = tData{:,3:end};          % predictors
vY       = tData{:,2};              % inflation target
cXnamesM = cAllNames(3:end)';       % predictor names as an Nx1 cell

%% Data preprocessing
% Predictive lag (periods). The predictors are already reporting-lag aligned
% in Python; this is the additional one-period predictive lag.
iTimeLag = 1;

% Determine dimensions
[iNumObs, iNumAssets]       = size(vY);
[iNumObsP, iNumPredictors]  = size(mX);

% Lag the predictors so that they are known one period ahead of the target
mXlag       = [NaN(iTimeLag, iNumPredictors); mX(1:end-iTimeLag,:)];

%% Settings
iNumIn = 60;                       % Number of in-sample periods (10 years)
iNumOut = 1;                        % Number of forecasting periods 
lRoll = true;                      % Rolling time window

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

% Loop over time. The in-sample window holds iNumIn observations and the
% forecast is made for the NEXT row (iIdxT+1). This matches the Python port
% (util.rolling_oos_forecast) and the bKap4 OOS convention: previously the
% rolling window held only iNumIn-1 rows and predicted iIdxT, which differed
% from the Python kitchen-sink script by one observation.
for iIdxT = iNumIn:iNumObsTemp-1
    % Get time indices
    if lRoll
        vIdxInSample = (iIdxT-iNumIn+1):iIdxT;
    else
        vIdxInSample = 1:iIdxT;
    end
    iIdxOut = iIdxT + 1;

    % Get data
    mXin    = mXtemp(vIdxInSample,:);
    vXout   = mXtemp(iIdxOut,:);
    vYin    = vYtemp(vIdxInSample,:);

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
    vYhatTemp(iIdxOut) = vXout * vBetaTemp;

    % Rolling mean prediction
    vYrollTemp(iIdxOut) = mean(vYin);
end

% Write forecasts back to their ORIGINAL calendar positions. The removed rows
% are scattered through the sample (a late start, a mid-series gap and trailing
% NaNs), not all at the front, so the compacted forecasts must be re-inserted by
% logical index. (Prepending sum(lIsNaN) NaNs would misalign every forecast with
% the actual series whenever a missing row sits anywhere but the start.)
mYhat  = NaN(iNumObs, 1);
mYroll = NaN(iNumObs, 1);
mYhat(~lIsNaN)  = vYhatTemp;
mYroll(~lIsNaN) = vYrollTemp;

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