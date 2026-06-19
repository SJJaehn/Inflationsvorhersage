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

%% In-Sample analysis
% Remove missing values
lIsNaN          = any(isnan(vXlag),2) | isnan(vY);
vY(lIsNaN,:)    = [];
vXlag(lIsNaN,:) = [];
dtDates(lIsNaN) = [];

% z-Transformation of predictor
vXlag = (vXlag - mean(vXlag))./std(vXlag);

% Regression with intercept
rResults = regstats2(vY, vXlag, 'linear', {'tstat','rsquare'});
vBeta    = rResults.tstat.beta;      % coefficient estimates
vBetaT   = rResults.tstat.t;         % t-Statistic estimates
dR2      = rResults.rsquare;         % R2

% Restore path
path(sOldPath);