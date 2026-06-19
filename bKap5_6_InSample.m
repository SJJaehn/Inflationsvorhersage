% Script for applying PLS dimensionality reduction to the Liedtke macro panel
% (in-sample). The macro predictors are reduced to a few PLS components (which,
% unlike PCA, are extracted to explain the inflation target) and inflation is
% regressed on those components.

% Clear console
clear; clc; close all;

% Set paths
sOldPath = path;
sDataPath = './DATA/Liedtke/US/';
sResultsPath = './RESULTS/GWZ/';
addpath('./Utils/');

%% Settings
iTimeLag    = 1;     % additional predictive lag (predictors are already
                     % reporting-lag aligned in DATA/Liedtke/aggregate.py)
iNumComp    = 3;     % number of PLS components to retain
iTransformX = 2;     % preprocessing: 0 = none, 1 = centre, 2 = z-standardise.
                     % The predictors live on very different scales (rate diffs
                     % vs percentage changes), so standardisation is used.

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

% Update number of observations after NaN removal
iNumObs  = size(vY,1);
iNumComp = min(iNumComp, iNumPredictors);

%% PLS dimensionality reduction
% Estimate PLS keeping iNumComp components
rModel  = fEstPLS(vY, mXlag, "iNumComp", iNumComp, "iTransformX", iTransformX);
mScores = rModel.mScores;

%% In-Sample analysis
% Regression with intercept using the retained PLS components
rResults = regstats2(vY, mScores, 'linear', {'tstat','rsquare'});
vBeta    = rResults.tstat.beta;   % coefficient estimates
vBetaT   = rResults.tstat.t;      % t-Statistics
dR2      = rResults.rsquare;      % R2

% In-sample prediction
vYhat = [ones(iNumObs,1), mScores] * vBeta;

% === Report
fprintf('In-sample PLS regression: inflation ~ %d PLS components\n', iNumComp);
fprintf('  Variance explained by %d comps : X %6.2f%% | Y %6.2f%%\n', iNumComp, ...
    rModel.mExplVar(1,end)*100, rModel.mExplVar(2,end)*100);
fprintf('  Intercept     : %8.4f (t = %6.2f)\n', vBeta(1), vBetaT(1));
for k = 1:iNumComp
    fprintf('  Comp%-2d coef   : %8.4f (t = %6.2f)\n', k, vBeta(k+1), vBetaT(k+1));
end
fprintf('  R2            : %8.4f\n', dR2);

% === Save results
sFilename = [sResultsPath, 'InSamplePLSResults.mat'];
save(sFilename, 'vBeta', 'vBetaT', 'dR2', 'vYhat', 'rModel');

% Restore path
path(sOldPath);
