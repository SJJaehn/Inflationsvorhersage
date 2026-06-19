% Script for performing the in-sample kitchen sink regression

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

%% In-Sample analysis
% Get data / copy data
mXtemp      = mXlag;
vYtemp      = vY;
dtDatesTemp = dtDates;

% Remove missing values
lIsNaN = isnan(vYtemp) | any(isnan(mXtemp),2);
vYtemp(lIsNaN)      = [];
mXtemp(lIsNaN,:)    = [];
dtDatesTemp(lIsNaN) = [];

% z-Transformation of predictor
mXtemp = (mXtemp - mean(mXtemp,1))./std(mXtemp,[],1);

% Regression with intercept
rResults    = regstats(vYtemp, mXtemp, 'linear', {'tstat','rsquare'});
vBeta       = rResults.tstat.beta;      % coefficient estimates
vBetaT      = rResults.tstat.t;         % t-Statistic estimates
dR2         = rResults.rsquare;         % R2

% Add constant
mXtemp = [ones(size(mXtemp,1),1), mXtemp];

% In-sample prediction
mYhat = mXtemp * vBeta;

% Add missing values
mYhat = [NaN(sum(lIsNaN),1); mYhat];

% === Create table
% Merge all results and round to two digits
mResults = round([vBeta(2:end)*100, vBetaT(2:end)],2);

% Make to cell
cTable = [cXnamesM, sprintfc('%.2f', mResults)];

% Add column header
cTable = [{'Predictor','Beta', 'Beta T'}; cTable];

% Add R2
cTable = [cTable; cell(1, size(cTable,2)); {'R2','',num2str(round(dR2 * 100,2))}];

% === Save results
sFilename = [sResultsPath,'InSampleKitchenSinkResults.mat'];
save(sFilename, "cTable", 'mYhat');

% Restore path
path(sOldPath);