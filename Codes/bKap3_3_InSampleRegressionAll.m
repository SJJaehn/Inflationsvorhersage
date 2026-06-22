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