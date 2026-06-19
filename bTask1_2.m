% Script for performing the in-sample regressions in GWZ

% Clear console
clear; clc; close all;

% Set paths
sOldPath = path;
sDataPath = './DATA/Empirical/';
sResultsPath = './RESULTS/GWZ/';
addpath('./Utils/');

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
vX          = [vY,mX(:,iIdxPred)];

%% Data preprocessing
% Determine dimensions
[iNumObs, iNumAssets]       = size(vY);
[iNumObsP, iNumPredictors]  = size(vX);

% Note: The predictor data and the market returns have the same time index.
% Therefore, we need to lag the predictor data so that we have a predictive
% regression
%vXlag = [NaN(1,iNumPredictors); vX(1:end-1,:)];

%% In-Sample analysis

% z-Transformation of predictor
% vXlag = (vXlag - mean(vXlag))./std(vXlag);

% Regression with intercept
rModelVAR1 = fEstVAR(vX, [], 'iNumLags', 1, 'lEstAlpha', true, 'lGetStats', true);

rModelVAR2 = fEstVAR(vX, [], 'iNumLags', 2, 'lEstAlpha', true);

% Restore path
path(sOldPath);