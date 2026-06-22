% Script for performing the in-sample kitchen sink regression

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

% Note: The predictor data and the market returns have the same time index.
% Therefore, we need to lag the predictor data so that we have a predictive
% regression
mXlag       = [NaN(1, iNumPredictors); mX(1:end-1,:)];

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