% Script for applying PCA dimensionality reduction to the Liedtke macro panel
% (in-sample). The macro predictors are reduced to a few principal components
% and the inflation target is regressed on those components (PCR).

% Clear console
clear; clc; close all;

% Set paths
sOldPath = path;
sCountry = fCfg('COUNTRY', 'US');
sDataPath = ['./DATA/Liedtke/', sCountry, '/'];
sResultsPath = './RESULTS/GWZ/';
addpath('./Utils/');

%% Settings
iTimeLag    = 1;     % additional predictive lag (predictors are already
                     % reporting-lag aligned in DATA/Liedtke/aggregate.py)
iNumComp    = 3;     % number of principal components to retain
iTransformX = 2;     % PCA preprocessing: 0 = none, 1 = centre, 2 = z-standardise.
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

% Remove missing values row-wise (PCA does not allow missing values)
lIsNaN          = isnan(vY) | any(isnan(mXlag),2);
vY(lIsNaN)      = [];
mXlag(lIsNaN,:) = [];
dtDates(lIsNaN) = [];

% Update number of observations after NaN removal
iNumObs  = size(vY,1);
iNumComp = min(iNumComp, iNumPredictors);

%% PCA dimensionality reduction
% Run full PCA to obtain explained variance for all factors
rModelFull = fEstPCA(mXlag, "iNumComp", iNumPredictors, "iTransformX", iTransformX);

% Plot explained variance per factor (individual, not cumulative)
vExplVarPerFactor = diff([0; rModelFull.vExplVar]) * 100;
figure;
bar(1:iNumPredictors, vExplVarPerFactor);
xlabel('Factor');
ylabel('Explained Variance (%)');
title('Explained Variance by Factor');
box off;

% Estimate PCA keeping iNumComp principal components
rModel  = fEstPCA(mXlag, "iNumComp", iNumComp, "iTransformX", iTransformX);
mScores = rModel.mScores;

%% In-Sample analysis
% Regression with intercept using the retained principal components
rResults = regstats2(vY, mScores, 'linear', {'tstat','rsquare'});
vBeta    = rResults.tstat.beta;   % coefficient estimates
vBetaT   = rResults.tstat.t;      % t-Statistics
dR2      = rResults.rsquare;      % R2

% In-sample prediction
vYhat = [ones(iNumObs,1), mScores] * vBeta;

% === Report
fprintf('In-sample PCA regression: inflation ~ %d principal components\n', iNumComp);
fprintf('  Variance explained by %d PCs : %6.2f%%\n', iNumComp, rModel.vExplVar(end)*100);
fprintf('  Intercept     : %8.4f (t = %6.2f)\n', vBeta(1), vBetaT(1));
for k = 1:iNumComp
    fprintf('  PC%-2d coef     : %8.4f (t = %6.2f)\n', k, vBeta(k+1), vBetaT(k+1));
end
fprintf('  R2            : %8.4f\n', dR2);

% === Save results
% Structured output: <GWZ>/PCA/<country>/insample/<options>/
sCountry = fCountryFromPath(sDataPath);
sOutDir  = fResultDir(sResultsPath, 'PCA', sCountry, 'insample', ...
    sprintf('comp%d_lag%d', iNumComp, iTimeLag));
exportgraphics(gcf, fullfile(sOutDir, 'chart.png'), 'Resolution', 150);
tRes = cell2table({vBeta(1), dR2}, 'VariableNames', {'Intercept', 'R2'});
writetable(tRes, fullfile(sOutDir, 'results.csv'));
sFilename = fullfile(sOutDir, 'results.mat');
save(sFilename, 'vBeta', 'vBetaT', 'dR2', 'vYhat', 'rModel');

% Restore path
path(sOldPath);
