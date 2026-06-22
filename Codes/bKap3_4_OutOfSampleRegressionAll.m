% Script for performing the out-of-sample regressions in GWZ

% Clear console
clear; clc; close all;

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

% Lag data
mXlag       = [NaN(1, iNumPredictors); mX(1:end-1,:)];

%% Settings
iNumIn = 240;                       % Number of in-sample periods (10 years)
iNumOut = 1;                        % Number of forecasting periods 
lRoll = false;                      % Rolling time window

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

% Compare with actual results
tResultsGWZ = readtable('./RESULTS/GWZ/ResultsGWZ.xlsx');

% Match
[cVars, idxA, idxB] = intersect(tResultsGWZ.Name, cXnamesM);

scatter(tResultsGWZ.OOSCT(idxA),rStatsOOS.vR2OOSCT(idxB)*100,'black','filled');
hold on
scatter(tResultsGWZ.OOSCT(idxA),rStatsOOS.vR2OOSCT(idxB)*100,'black','filled');
x_line = linspace(min(xlim),max(xlim),10);
plot(x_line,x_line,'black')
hold off
% title('Replikation der out-of-sample Ergebnisse','FontSize',12)
ylabel('Replicated $R^2$ (in \%)','FontSize',12,'Interpreter','latex');
xlabel('GWZ $R^2$ (in \%)','FontSize',12,'Interpreter','latex');
box off


% Restore path
path(sOldPath);