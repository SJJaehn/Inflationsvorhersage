% inflation_prediction_single.m
% Rolling/expanding-window OOS inflation prediction for each predictor
% individually (one step ahead). Results are saved to a CSV in the output
% directory.
%
% At every step a predictor is only used if all of its training-window
% values and the next test-step value exist (and the training target is
% complete). Otherwise that step is skipped.
%
% CSV format: col 1 = date, col 2 = target, col 3..end = predictors.

clear; clc; close all;

% =========================================================================
%  CONFIG — edit here
% =========================================================================
sCountry    = fCfg('COUNTRY', 'US');                 % 'US' or 'UK'
sCSVPath    = ['./DATA/Liedtke/', sCountry, '/aggregated.csv'];    % input CSV
sOutputDir  = './RESULTS/';                     % output directory

lRolling         = fCfg('ROLLING', true);   % true = rolling window, false = expanding window
iTrainObs        = 60;     % in-sample window length (number of observations)
iTimeLag         = 1;      % predictor lag (periods; 1 = standard predictive regression)
lSharedTimeframe = true;  % true = evaluate every predictor on the SAME sample
                           %        (only rows where target & ALL predictors exist)
% =========================================================================

% Add utilities
sOldPath = path;
addpath('./Utils/');

% =========================================================================
%  1. Load data
% =========================================================================
tData     = readtable(sCSVPath);
cAllNames = tData.Properties.VariableNames;   % all column names

% Parse dates (first column)
dtDates = tData{:,1};
if ~isdatetime(dtDates)
    dtDates = datetime(dtDates);
end

% Target: second column; predictors: columns 3..end
vY            = tData{:,2};
mX_raw        = tData{:,3:end};
cPredNames    = cAllNames(3:end);
iNumPredictors = size(mX_raw, 2);
iNumObs       = length(vY);

fprintf('Loaded %d observations, %d predictors from %s\n', ...
    iNumObs, iNumPredictors, sCSVPath);

% =========================================================================
%  2. Apply time lag to all predictors
% =========================================================================
% mXlag(t) contains the predictor value from t-iTimeLag, so that it is
% available at time t and can predict vY(t).
if iTimeLag > 0
    mXlag = [NaN(iTimeLag, iNumPredictors); mX_raw(1:end-iTimeLag, :)];
else
    mXlag = mX_raw;
end

% Optionally restrict to the shared timeframe: keep only rows where the
% target AND all predictors have data, so every predictor is evaluated on
% the exact same observations (comparable to the forward-backward script).
if lSharedTimeframe
    lComplete = ~isnan(vY) & ~any(isnan(mXlag), 2);
    vY      = vY(lComplete);
    mXlag   = mXlag(lComplete, :);
    dtDates = dtDates(lComplete);
    iNumObs = length(vY);
    fprintf('Shared timeframe: %d of %d observations retained.\n', ...
        iNumObs, numel(lComplete));
end

% =========================================================================
%  3. Loop over predictors
% =========================================================================
cResults = cell(iNumPredictors, 1);   % one struct per predictor

for iIdxP = 1:iNumPredictors

    sPredName = cPredNames{iIdxP};
    fprintf('Processing predictor %d/%d: %s\n', iIdxP, iNumPredictors, sPredName);

    vXcol = mXlag(:, iIdxP);

    % --- Rolling/expanding one-step-ahead prediction loop ---------------
    vYhat   = NaN(iNumObs, 1);    % model forecast
    vYhatBM = NaN(iNumObs, 1);    % benchmark: historical mean

    for iIdxT = iTrainObs : iNumObs - 1
        % In-sample index
        if lRolling
            vIdxIn = (iIdxT - iTrainObs + 1) : iIdxT;
        else
            vIdxIn = 1 : iIdxT;
        end
        iIdxOut = iIdxT + 1;            % one step ahead

        vYin  = vY(vIdxIn);
        vXin  = vXcol(vIdxIn);
        vXout = vXcol(iIdxOut);

        % Use this step only if all training and next-step data exist
        if any(isnan(vXin)) || any(isnan(vYin)) || isnan(vXout)
            continue
        end

        % OLS with constant: [1, x] * beta = y
        mXin_reg  = [ones(length(vIdxIn), 1), vXin];
        mXout_reg = [1, vXout];
        vBeta     = mXin_reg \ vYin;

        vYhat(iIdxOut)   = mXout_reg * vBeta;
        vYhatBM(iIdxOut) = mean(vYin);      % historical mean benchmark
    end

    % --- Metrics using fEvaluatePerformanceOOS --------------------------
    rOOS = fEvaluatePerformanceOOS(vY, vYhatBM, vYhat);

    % --- Metrics using fForecastQuality (RMSE, MAE, etc.) ---------------
    lValid = ~isnan(vY) & ~isnan(vYhat);
    if sum(lValid) >= 3
        rQ = fForecastQuality(vY(lValid), vYhat(lValid));
        dRMSE     = rQ.vRMSE;
        dMAE      = rQ.vMAE;
        dCor      = rQ.vCor;
        dHitRate  = rQ.vHitRate;
        dR2_MZ    = rQ.rRegResults.vR2;
        dF_MZ     = rQ.rRegResults.vF;
        dP_MZ     = rQ.rRegResults.vP;
    else
        dRMSE = NaN; dMAE = NaN; dCor = NaN;
        dHitRate = NaN; dR2_MZ = NaN; dF_MZ = NaN; dP_MZ = NaN;
    end

    % --- OOS start / end dates ------------------------------------------
    iFirstOOS = find(lValid, 1, 'first');
    iLastOOS  = find(lValid, 1, 'last');
    if isempty(iFirstOOS)
        dtOOSBeg = NaT; dtOOSEnd = NaT;
    else
        dtOOSBeg = dtDates(iFirstOOS);
        dtOOSEnd = dtDates(iLastOOS);
    end

    % --- Store results --------------------------------------------------
    cResults{iIdxP} = struct( ...
        'Name',        sPredName, ...
        'OOSBeg',      dtOOSBeg, ...
        'OOSEnd',      dtOOSEnd, ...
        'NumOOSObs',   sum(lValid), ...
        'R2OOS',       rOOS.vR2OOS, ...
        'R2OOS_CT',    rOOS.vR2OOSCT, ...
        'CW_stat',     rOOS.vCW, ...
        'CW_p',        rOOS.vCWp, ...
        'CW_stat_CT',  rOOS.vCW_CT, ...
        'CW_p_CT',     rOOS.vCWp_CT, ...
        'DM_stat',     rOOS.vDM, ...
        'DM_p',        rOOS.vDMp, ...
        'RMSE',        dRMSE, ...
        'MAE',         dMAE, ...
        'Cor',         dCor, ...
        'HitRate',     dHitRate, ...
        'R2_MZ',       dR2_MZ, ...
        'F_MZ',        dF_MZ, ...
        'p_MZ',        dP_MZ ...
    );

    fprintf('  -> %d OOS obs | OOS R2=%.4f | R2-CT=%.4f | RMSE=%.4f | MAE=%.4f | Cor=%.4f\n', ...
        sum(lValid), rOOS.vR2OOS, rOOS.vR2OOSCT, dRMSE, dMAE, dCor);
end

% =========================================================================
%  4. Assemble and save results table
% =========================================================================
cRowData  = {};
cColNames = {'Predictor','OOS_Beg','OOS_End','Num_OOS_Obs', ...
    'R2_OOS','R2_OOS_CT','CW_stat','CW_p','CW_stat_CT','CW_p_CT', ...
    'DM_stat','DM_p','RMSE','MAE','Cor','HitRate','R2_MZ','F_MZ','p_MZ'};

for iIdxP = 1:iNumPredictors
    r = cResults{iIdxP};
    if isempty(r)
        continue
    end

    % Date strings (empty if no valid OOS observation)
    if isnat(r.OOSBeg)
        sBeg = ''; sEnd = '';
    else
        sBeg = char(r.OOSBeg, 'yyyy-MM-dd');
        sEnd = char(r.OOSEnd, 'yyyy-MM-dd');
    end

    cRowData(end+1, :) = { ...
        r.Name, sBeg, sEnd, r.NumOOSObs, ...
        r.R2OOS, r.R2OOS_CT, r.CW_stat, r.CW_p, r.CW_stat_CT, r.CW_p_CT, ...
        r.DM_stat, r.DM_p, ...
        r.RMSE, r.MAE, r.Cor, r.HitRate, r.R2_MZ, r.F_MZ, r.p_MZ ...
    }; %#ok<AGROW>
end

if isempty(cRowData)
    warning('No valid results to save.');
else
    tOut = cell2table(cRowData, 'VariableNames', cColNames);

    % Structured output dir: <root>/single/<country>/oos/<options>/results.csv
    sCountry = fCountryFromPath(sCSVPath);
    if lRolling, sWindow = 'rolling'; else, sWindow = 'expanding'; end
    sOptions = sprintf('train%d_%s_lag%d', iTrainObs, sWindow, iTimeLag);
    sOutDir  = fResultDir(sOutputDir, 'single', sCountry, 'oos', sOptions);
    sOutFile = fullfile(sOutDir, 'results.csv');
    writetable(tOut, sOutFile);
    fprintf('\nResults saved to: %s\n', sOutFile);

    % Print summary table to console
    fprintf('\n========== Summary ===========\n');
    disp(tOut(:, {'Predictor','R2_OOS','R2_OOS_CT','CW_p','CW_p_CT','RMSE','MAE','Cor'}));
end

% Restore path
path(sOldPath);
