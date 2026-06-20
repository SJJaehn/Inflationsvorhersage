% Script for performing the out-of-sample regressions in GWZ

% Clear console
clear; clc; close all;

% Clear console
clear; clc; close all;

% Set paths
sOldPath = path;
sDataPath = './DATA/Liedtke/US/';
sResultsPath = './RESULTS/GWZ/';
addpath('./Utils/');

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
% Initialize memory
mYhat           = NaN(iNumObs, iNumPredictors); % Predicted values
mYroll          = NaN(iNumObs, iNumPredictors); % Rolling mean forecast
dtDateSample    = NaT(iNumPredictors, 2);       % Start and end of sample

for iIdxP = 1:iNumPredictors
    % Print progress
    fprintf('Estimate in-sample %s\n', cXnamesM{iIdxP})

    % Get data
    vXtemp      = mXlag(:,iIdxP);
    vYtemp      = vY;
    dtDateTemp  = dtDates;

    % Remove missing values
    lIsNaN = isnan(vXtemp) | isnan(vYtemp);
    vYtemp(lIsNaN)      = [];
    vXtemp(lIsNaN)      = [];
    dtDateTemp(lIsNaN)  = [];

    % Save start and end date
    dtDateSample(iIdxP, :) = [dtDateTemp(1+iNumIn);dtDateTemp(end)]; % The first colum is now start of OOS date

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
        vXin    = vXtemp(vIdxInSample,:);       % t-1
        vXout   = vXtemp(iIdxT,:);              % t-1
        vYin    = vYtemp(vIdxInSample,:);       % t

        % Regress only if all observations are available
        if any(isnan(vXin))
            continue
        end

        % Add constant
        mXin    = [ones(size(vXin,1),1), vXin];
        mXout   = [ones(size(vXout,1),1), vXout];

        % Regression
        vBetaTemp = mXin\vYin;

        % Prediction
        vYhatTemp(iIdxT) = mXout * vBetaTemp;

        % Rolling mean prediction
        vYrollTemp(iIdxT) = mean(vYin);
    end   

    % Save predictions in the full matrix
    mYhat(~lIsNaN,iIdxP) = vYhatTemp;
    mYroll(~lIsNaN,iIdxP) = vYrollTemp;
end

% Evaluate quality
[rStatsOOS] = fEvaluatePerformanceOOS(vY,mYroll,mYhat);

% === Create table
% Merge all results and round to two digits
mResults = round([rStatsOOS.vR2OOS * 100; ...
    rStatsOOS.vCWp * 100; ...
    rStatsOOS.vR2OOSCT * 100;...
    rStatsOOS.vCWp_CT * 100],2);

% Make to cell
cTable3_OOS = [cXnamesM, sprintfc('%.2f', mResults')];

% Add column header
cTable3_OOS = [{'Predictor','OOS R2', 'CW (p)', 'OOS R2 CT', 'CW (p) CT'}; cTable3_OOS];

% === Save results
sFilename = [sResultsPath,'OutOfSampleResults.mat'];
save(sFilename, "cTable3_OOS", 'cXnamesM','mYhat','mYroll','rStatsOOS');

% NOTE: The original GWZ replication comparison (vs ./RESULTS/GWZ/ResultsGWZ.xlsx)
% was removed because it matches GWZ predictor names, which do not exist in this
% inflation dataset. Bar chart of OOS R2 (in %) per predictor instead:
bar(rStatsOOS.vR2OOSCT * 100);
set(gca, 'XTick', 1:iNumPredictors, 'XTickLabel', cXnamesM, 'TickLabelInterpreter', 'none');
xtickangle(45);
ylabel('OOS $R^2$ CT (in \%)','FontSize',12,'Interpreter','latex');
box off


% Restore path
path(sOldPath);