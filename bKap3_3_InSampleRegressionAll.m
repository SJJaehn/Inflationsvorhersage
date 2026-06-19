% Script for performing the in-sample regressions in GWZ

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

%% In-Sample analysis
% Initialize memory
mYhat           = NaN(iNumObs, iNumPredictors);     % Predicted values
dtDateSample    = NaT(iNumPredictors, 2);           % Start and end date of analysis
mBeta           = NaN(iNumPredictors, 2);           % Beta coefficients
mBetaT          = NaN(iNumPredictors, 2);           % t-Statistics of beta coefficients
vR2             = NaN(iNumPredictors, 1);           % R2

% Perform in-sample regression for each predictor
for iIdxP = 1:iNumPredictors
    % Print progress
    fprintf('Estimate in-sample %s\n', cXnamesM{iIdxP})

    % Get data / copy data
    vXtemp      = mXlag(:,iIdxP);
    vYtemp      = vY;
    dtDatesTemp = dtDates;

    % Remove missing values
    lIsNaN = isnan(vXtemp) | isnan(vYtemp);
    vYtemp(lIsNaN)      = [];
    vXtemp(lIsNaN)      = [];
    dtDatesTemp(lIsNaN) = [];

    % Save start and end date
    dtDateSample(iIdxP, :) = [dtDatesTemp(1) , dtDatesTemp(end)];

    % z-Transformation of predictor
    vXtemp = (vXtemp - mean(vXtemp))./std(vXtemp);

    % Regression with intercept
    rResults    = regstats(vYtemp, vXtemp, 'linear', {'tstat','rsquare'});
    vBetaTemp   = rResults.tstat.beta;      % coefficient estimates
    vBetaTempT  = rResults.tstat.t;         % t-Statistic estimates
    dR2         = rResults.rsquare;         % R2

    % Add constant
    mXtemp = [ones(size(vXtemp,1),1), vXtemp];

    % In-sample prediction
    mYhat(~lIsNaN,iIdxP)  = mXtemp * vBetaTemp;

    % Save beta, t-statistics, and R2
    mBeta(iIdxP,:)    = vBetaTemp; 
    mBetaT(iIdxP,:)   = vBetaTempT; 
    vR2(iIdxP)        = dR2;
end

% === Create table
% Merge all results and round to two digits
mResults = round([mBeta(:,2)*100, mBetaT(:,2), vR2 * 100],2);

% Make to cell
cTable3 = [cXnamesM, sprintfc('%.2f', mResults)];

% Add column header
cTable3 = [{'Predictor','Beta', 'Beta T', 'R2'}; cTable3];

% === Save results
sFilename = [sResultsPath,'InSampleResults.mat'];
save(sFilename, "cTable3", 'mBetaT','mBeta','vR2','cXnamesM','mYhat');

% % Compare with actual results
% tResultsGWZ = readtable('./RESULTS/GWZ/ResultsGWZ.xlsx');
% 
% % Match
% [cVars, idxA, idxB] = intersect(tResultsGWZ.Name, cXnamesM);
% scatter(tResultsGWZ.ISR2(idxA),vR2(idxB)*100,'black','filled');
% lsline
% % title('Replikation der in-sample Ergebnisse','FontSize',12)
% ylabel('Replicated $R^2$ (in \%)','FontSize',12,'Interpreter','latex');
% xlabel('GWZ $R^2$ (in \%)','FontSize',12,'Interpreter','latex');
% box off



% Restore path
path(sOldPath);