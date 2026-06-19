% inflation_prediction_full.m
% "Kitchen-sink" one-step-ahead OOS inflation forecast: a single regression
% that uses ALL predictors at once, re-estimated on a rolling/expanding
% window. Benchmarked against the historical-mean forecast.
%
% Modelled on bKap3_6_OutOfSampleKitchenSink.m, but with the project's CSV
% layout and per-step data-availability handling.
%
% At every origin t the model is used only if the whole training window and
% the next test step have complete data for the target and every predictor.
%
% CSV format: col 1 = date, col 2 = target, col 3..end = predictors.

clear; clc; close all;

% =========================================================================
%  CONFIG — edit here
% =========================================================================
sCSVPath    = './DATA/Liedtke/US/aggregated.csv';  % input CSV
sOutputDir  = './RESULTS/';                     % output directory

lRolling    = true;     % true = rolling window, false = expanding window
iTrainObs   = 60;      % in-sample window length (number of observations)
iTimeLag    = 1;        % predictor lag (periods; 1 = standard predictive regression)

% Predictors to use. Leave empty ({}) to use ALL columns 3..end. Otherwise
% give a cell array of predictor (column) names, e.g.:
%   cUsePredictors = {'unemp', 'oil', 'm2'};
cUsePredictors = {};
% =========================================================================

% Add utilities
sOldPath = path;
addpath('./Utils/');

% =========================================================================
%  1. Load data
% =========================================================================
tData     = readtable(sCSVPath);
cAllNames = tData.Properties.VariableNames;

dtDates = tData{:,1};
if ~isdatetime(dtDates)
    dtDates = datetime(dtDates);
end

vY          = tData{:,2};
mX_raw      = tData{:,3:end};
cPredNames  = cAllNames(3:end);
iNumObs     = length(vY);

% --- Optional subset of predictors (kept in the order listed) -----------
if ~isempty(cUsePredictors)
    [lFound, vLoc] = ismember(cUsePredictors, cPredNames);
    if ~all(lFound)
        error('Predictor(s) not found in CSV: %s', ...
            strjoin(cUsePredictors(~lFound), ', '));
    end
    mX_raw     = mX_raw(:, vLoc);       % vLoc follows cUsePredictors order
    cPredNames = cPredNames(vLoc);
end
iNumPred = size(mX_raw, 2);

fprintf('Loaded %d observations; using %d of %d predictors from %s\n', ...
    iNumObs, iNumPred, numel(cAllNames)-2, sCSVPath);
fprintf('Predictors: %s\n', strjoin(cPredNames, ', '));

% =========================================================================
%  2. Apply time lag
% =========================================================================
% mXlag(t) holds the predictor value from t-iTimeLag, so it is available at
% time t and can predict vY(t).
if iTimeLag > 0
    mXlag = [NaN(iTimeLag, iNumPred); mX_raw(1:end-iTimeLag, :)];
else
    mXlag = mX_raw;
end

% =========================================================================
%  3. Rolling/expanding one-step-ahead loop (all predictors at once)
% =========================================================================
vYhat   = NaN(iNumObs, 1);    % full-model forecast
vYhatBM = NaN(iNumObs, 1);    % benchmark: historical mean

for iIdxT = iTrainObs : iNumObs - 1
    % In-sample window
    if lRolling
        vIn = (iIdxT - iTrainObs + 1) : iIdxT;
    else
        vIn = 1 : iIdxT;
    end
    iOut = iIdxT + 1;            % one step ahead

    vYin  = vY(vIn);
    mXin  = mXlag(vIn, :);
    vXout = mXlag(iOut, :);

    % Use this step only if all training and next-step data exist
    if any(isnan(vYin)) || any(isnan(mXin(:))) || any(isnan(vXout))
        continue
    end

    % OLS with constant on the full predictor set
    mXin_r  = [ones(numel(vIn), 1), mXin];
    mXout_r = [1, vXout];
    vBeta   = mXin_r \ vYin;

    vYhat(iOut)   = mXout_r * vBeta;
    vYhatBM(iOut) = mean(vYin);     % historical-mean benchmark
end

% =========================================================================
%  4. Metrics
% =========================================================================
rOOS = fEvaluatePerformanceOOS(vY, vYhatBM, vYhat);

lValid = ~isnan(vY) & ~isnan(vYhat);
if sum(lValid) >= 3
    rQ      = fForecastQuality(vY(lValid), vYhat(lValid));
    dRMSE   = rQ.vRMSE;
    dMAE    = rQ.vMAE;
    dCor    = rQ.vCor;
    dHit    = rQ.vHitRate;
    dR2_MZ  = rQ.rRegResults.vR2;
    dF_MZ   = rQ.rRegResults.vF;
    dP_MZ   = rQ.rRegResults.vP;
else
    dRMSE = NaN; dMAE = NaN; dCor = NaN;
    dHit = NaN; dR2_MZ = NaN; dF_MZ = NaN; dP_MZ = NaN;
end

% OOS date range
iFirst = find(lValid, 1, 'first');
iLast  = find(lValid, 1, 'last');
if isempty(iFirst)
    dtOOSBeg = NaT; dtOOSEnd = NaT; sBeg = ''; sEnd = '';
else
    dtOOSBeg = dtDates(iFirst); dtOOSEnd = dtDates(iLast);
    sBeg = char(dtOOSBeg, 'yyyy-MM-dd'); sEnd = char(dtOOSEnd, 'yyyy-MM-dd');
end

fprintf('========== Full (all-predictor) model metrics ===========\n');
fprintf('  Predictors   : %d\n', iNumPred);
fprintf('  OOS obs      : %d\n', sum(lValid));
fprintf('  OOS R2       : %.6f\n', rOOS.vR2OOS);
fprintf('  OOS R2-CT    : %.6f\n', rOOS.vR2OOSCT);
fprintf('  CW stat / p  : %.4f / %.4f\n', rOOS.vCW, rOOS.vCWp);
fprintf('  CW-CT stat/p : %.4f / %.4f\n', rOOS.vCW_CT, rOOS.vCWp_CT);
fprintf('  DM stat / p  : %.4f / %.4f\n', rOOS.vDM, rOOS.vDMp);
fprintf('  RMSE         : %.6f\n', dRMSE);
fprintf('  MAE          : %.6f\n', dMAE);
fprintf('  Correlation  : %.6f\n', dCor);
fprintf('  Hit rate     : %.6f\n', dHit);
fprintf('  MZ R2        : %.6f\n', dR2_MZ);
fprintf('  MZ F / p     : %.4f / %.4f\n', dF_MZ, dP_MZ);
if ~isempty(sBeg)
    fprintf('  OOS period   : %s  to  %s\n', sBeg, sEnd);
end

% =========================================================================
%  5. Save results
% =========================================================================
if ~exist(sOutputDir, 'dir')
    mkdir(sOutputDir);
end
sTimestamp = datestr(now, 'yyyymmdd_HHMMSS');

if lRolling; sWindowType = 'rolling'; else; sWindowType = 'expanding'; end

% --- (a) Summary ---------------------------------------------------------
cSummary = { ...
    'WindowType',   sWindowType; ...
    'TrainObs',     num2str(iTrainObs); ...
    'TimeLag',      num2str(iTimeLag); ...
    'NumPredictors',num2str(iNumPred); ...
    'OOS_Beg',      sBeg; ...
    'OOS_End',      sEnd; ...
    'Num_OOS_Obs',  num2str(sum(lValid)); ...
    'R2_OOS',       num2str(rOOS.vR2OOS,  '%.6f'); ...
    'R2_OOS_CT',    num2str(rOOS.vR2OOSCT,'%.6f'); ...
    'CW_stat',      num2str(rOOS.vCW,     '%.4f'); ...
    'CW_p',         num2str(rOOS.vCWp,    '%.4f'); ...
    'CW_stat_CT',   num2str(rOOS.vCW_CT,  '%.4f'); ...
    'CW_p_CT',      num2str(rOOS.vCWp_CT, '%.4f'); ...
    'DM_stat',      num2str(rOOS.vDM,     '%.4f'); ...
    'DM_p',         num2str(rOOS.vDMp,    '%.4f'); ...
    'RMSE',         num2str(dRMSE,  '%.6f'); ...
    'MAE',          num2str(dMAE,   '%.6f'); ...
    'Cor',          num2str(dCor,   '%.6f'); ...
    'HitRate',      num2str(dHit,   '%.6f'); ...
    'R2_MZ',        num2str(dR2_MZ, '%.6f'); ...
    'F_MZ',         num2str(dF_MZ,  '%.4f'); ...
    'p_MZ',         num2str(dP_MZ,  '%.4f'); ...
};
tSummary = cell2table(cSummary, 'VariableNames', {'Key','Value'});
sSumFile = fullfile(sOutputDir, ['full_summary_', sTimestamp, '.csv']);
writetable(tSummary, sSumFile);
fprintf('\nSummary saved to:     %s\n', sSumFile);

% --- (b) Predictions -----------------------------------------------------
tPred = table(dtDates, vY, vYhat, vYhatBM, ...
    'VariableNames', {'Date','Actual','Forecast','Benchmark'});
sPredFile = fullfile(sOutputDir, ['full_predictions_', sTimestamp, '.csv']);
writetable(tPred, sPredFile);
fprintf('Predictions saved to: %s\n', sPredFile);

% Restore path
path(sOldPath);
