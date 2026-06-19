% inflation_prediction_forward_backward.m
% Forward-backward stepwise predictor selection using rolling/expanding
% OOS regression. At each step the algorithm tries to add the predictor
% that most improves the chosen metric, then tries to remove any predictor
% whose removal would further improve it. Repeats until stable.
%
% CSV format: col 1 = date, col 2 = target, col 3..end = predictors.

clear; clc; close all;

% =========================================================================
%  CONFIG — edit here
% =========================================================================
sCSVPath    = './DATA/Liedtke/US/aggregated.csv';    % input CSV
sOutputDir  = './RESULTS/';                     % output directory

lRolling    = true;     % true = rolling window, false = expanding window
iTrainObs   = 60;       % in-sample window length (number of observations)
iTimeLag    = 1;        % predictor lag (periods; 1 = standard predictive regression)

% Selection metric — choose ONE of:
%   'R2OOS'    OOS R² vs. historical-mean benchmark  (higher is better)
%   'R2OOS_CT' OOS R² with Campbell-Thompson adj.    (higher is better)
%   'RMSE'     Root mean squared error               (lower  is better)
%   'MAE'      Mean absolute error                   (lower  is better)
%   'Cor'      Pearson correlation                   (higher is better)
%   'HitRate'  Directional accuracy                  (higher is better)
sMetric = 'RMSE';
% =========================================================================

% Map metric → direction
cHigherBetter = {'R2OOS','R2OOS_CT','Cor','HitRate'};
lHigherIsBetter = ismember(sMetric, cHigherBetter);

% "Worst possible" sentinel used to initialise comparisons
if lHigherIsBetter
    dWorst = -Inf;
else
    dWorst =  Inf;
end

% Add utilities
sOldPath = path;
addpath('./Utils/');

% =========================================================================
%  1. Load data
% =========================================================================
tData     = readtable(sCSVPath);
cAllNames = tData.Properties.VariableNames;

dtDates_raw = tData{:,1};
if ~isdatetime(dtDates_raw)
    dtDates_raw = datetime(dtDates_raw);
end

vY_raw       = tData{:,2};
mX_raw       = tData{:,3:end};
cPredNames   = cAllNames(3:end);
iNumPred     = size(mX_raw, 2);
iNumObs_raw  = length(vY_raw);

fprintf('Loaded %d observations, %d predictors from %s\n', ...
    iNumObs_raw, iNumPred, sCSVPath);
if lHigherIsBetter; sDir = 'higher'; else; sDir = 'lower'; end
fprintf('Metric: %s  (%s is better)\n\n', sMetric, sDir);

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

% Restrict to a common complete sample: keep only rows where the target and
% ALL predictors have data. This guarantees every candidate predictor set is
% evaluated on the exact same observations (fair metric comparison).
lComplete = ~isnan(vY_raw) & ~any(isnan(mXlag), 2);
vY        = vY_raw(lComplete);
mXlag     = mXlag(lComplete, :);
dtDates   = dtDates_raw(lComplete);
iNumObs   = length(vY);

fprintf('Common complete sample: %d of %d observations retained.\n\n', ...
    iNumObs, iNumObs_raw);

% =========================================================================
%  3. Helper: run OOS loop for a given set of predictor column indices
%     Returns the scalar metric value and the full prediction vector.
% =========================================================================
function [dMetricVal, vYhat, vYhatBM] = fRunOOS(vY_, mXlag_, vIdxSel, ...
        iTrainObs_, lRolling_, sMetric_, lHigherIsBetter_)

    iN   = length(vY_);
    iK   = length(vIdxSel);   % number of selected predictors

    vYhat   = NaN(iN, 1);
    vYhatBM = NaN(iN, 1);

    for iT = iTrainObs_ : iN - 1
        if lRolling_
            vIn = (iT - iTrainObs_ + 1) : iT;
        else
            vIn = 1 : iT;
        end
        iOut = iT + 1;

        vYin  = vY_(vIn);

        % Training target must be complete to use this step
        if any(isnan(vYin))
            continue
        end

        if iK == 0
            % No predictors: forecast = in-sample mean
            vYhat(iOut)   = mean(vYin);
            vYhatBM(iOut) = mean(vYin);
            continue
        end

        mXin  = mXlag_(vIn,  vIdxSel);
        vXout = mXlag_(iOut, vIdxSel);

        % Use this step only if all training and next-step predictor data exist
        if any(isnan(mXin(:))) || any(isnan(vXout))
            continue
        end

        mXin_r  = [ones(length(vIn), 1), mXin];
        mXout_r = [1, vXout];
        vBeta   = mXin_r \ vYin;

        vYhat(iOut)   = mXout_r * vBeta;
        vYhatBM(iOut) = mean(vYin);
    end

    % Compute the scalar metric
    lValid = ~isnan(vY_) & ~isnan(vYhat);
    if sum(lValid) < 3
        if lHigherIsBetter_; dMetricVal = -Inf; else; dMetricVal = Inf; end
        return
    end

    switch sMetric_
        case {'R2OOS','R2OOS_CT'}
            rOOS = fEvaluatePerformanceOOS(vY_, vYhatBM, vYhat);
            if strcmp(sMetric_,'R2OOS')
                dMetricVal = rOOS.vR2OOS;
            else
                dMetricVal = rOOS.vR2OOSCT;
            end
        case {'RMSE','MAE','Cor','HitRate'}
            rQ = fForecastQuality(vY_(lValid), vYhat(lValid));
            switch sMetric_
                case 'RMSE';    dMetricVal = rQ.vRMSE;
                case 'MAE';     dMetricVal = rQ.vMAE;
                case 'Cor';     dMetricVal = rQ.vCor;
                case 'HitRate'; dMetricVal = rQ.vHitRate;
            end
        otherwise
            error('Unknown metric: %s', sMetric_);
    end
end

% =========================================================================
%  4. Forward-backward stepwise selection
% =========================================================================
vSelected   = [];       % indices of currently selected predictors
cStepLog    = {};       % log of every add/remove decision

% Metric of the empty model (historical mean only)
[dCurrent, ~, ~] = fRunOOS(vY, mXlag, [], iTrainObs, lRolling, sMetric, lHigherIsBetter);
fprintf('Empty model %s = %.6f\n\n', sMetric, dCurrent);

iStep    = 0;
lChanged = true;

while lChanged
    lChanged = false;
    iStep    = iStep + 1;
    fprintf('=== Iteration %d ===\n', iStep);

    % ------------------------------------------------------------------
    %  Forward step: try adding each predictor not yet selected
    % ------------------------------------------------------------------
    vCandidates = setdiff(1:iNumPred, vSelected);
    dBestAdd    = dCurrent;
    iBestAddIdx = -1;

    for iIdxP = vCandidates
        vTry = sort([vSelected, iIdxP]);
        [dTry, ~, ~] = fRunOOS(vY, mXlag, vTry, iTrainObs, lRolling, sMetric, lHigherIsBetter);
        if (lHigherIsBetter && dTry > dBestAdd) || (~lHigherIsBetter && dTry < dBestAdd)
            dBestAdd    = dTry;
            iBestAddIdx = iIdxP;
        end
    end

    if iBestAddIdx > 0
        vSelected = sort([vSelected, iBestAddIdx]);
        dCurrent  = dBestAdd;
        lChanged  = true;
        sAdded    = cPredNames{iBestAddIdx};
        fprintf('  ADD    %-20s  ->  %s = %.6f\n', sAdded, sMetric, dCurrent);
        cStepLog(end+1,:) = {iStep, 'ADD', sAdded, dCurrent}; %#ok<AGROW>
    else
        fprintf('  No improvement from adding any predictor.\n');
    end

    % ------------------------------------------------------------------
    %  Backward step: try removing each currently selected predictor
    % ------------------------------------------------------------------
    lRemovedAny = true;
    while lRemovedAny && ~isempty(vSelected)
        lRemovedAny = false;
        dBestRem    = dCurrent;
        iBestRemIdx = -1;

        for iIdxR = 1:length(vSelected)
            vTry = vSelected;
            vTry(iIdxR) = [];
            [dTry, ~, ~] = fRunOOS(vY, mXlag, vTry, iTrainObs, lRolling, sMetric, lHigherIsBetter);
            if (lHigherIsBetter && dTry > dBestRem) || (~lHigherIsBetter && dTry < dBestRem)
                dBestRem    = dTry;
                iBestRemIdx = iIdxR;
            end
        end

        if iBestRemIdx > 0
            sRemoved        = cPredNames{vSelected(iBestRemIdx)};
            vSelected(iBestRemIdx) = [];
            dCurrent        = dBestRem;
            lChanged        = true;
            lRemovedAny     = true;
            fprintf('  REMOVE %-20s  ->  %s = %.6f\n', sRemoved, sMetric, dCurrent);
            cStepLog(end+1,:) = {iStep, 'REMOVE', sRemoved, dCurrent}; %#ok<AGROW>
        end
    end

    if ~lChanged
        fprintf('  No further improvement — stopping.\n');
    end
    fprintf('\n');
end

% =========================================================================
%  5. Final model: compute all metrics on the selected predictor set
% =========================================================================
fprintf('=== Final selected predictors (%d) ===\n', length(vSelected));
if isempty(vSelected)
    fprintf('  (none — empty model is best)\n');
else
    fprintf('  %s\n', strjoin(cPredNames(vSelected), ', '));
end
fprintf('\n');

[~, vYhat_final, vYhatBM_final] = fRunOOS(vY, mXlag, vSelected, iTrainObs, lRolling, sMetric, lHigherIsBetter);

lValid = ~isnan(vY) & ~isnan(vYhat_final);

% OOS metrics
rOOS = fEvaluatePerformanceOOS(vY, vYhatBM_final, vYhat_final);

% fForecastQuality metrics
if sum(lValid) >= 3
    rQ      = fForecastQuality(vY(lValid), vYhat_final(lValid));
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
    dtOOSBeg = NaT; dtOOSEnd = NaT;
else
    dtOOSBeg = dtDates(iFirst);
    dtOOSEnd = dtDates(iLast);
end

fprintf('Final model metrics:\n');
fprintf('  OOS R²       : %.6f\n', rOOS.vR2OOS);
fprintf('  OOS R²-CT    : %.6f\n', rOOS.vR2OOSCT);
fprintf('  CW stat / p  : %.4f / %.4f\n', rOOS.vCW, rOOS.vCWp);
fprintf('  CW-CT stat/p : %.4f / %.4f\n', rOOS.vCW_CT, rOOS.vCWp_CT);
fprintf('  DM stat / p  : %.4f / %.4f\n', rOOS.vDM, rOOS.vDMp);
fprintf('  RMSE         : %.6f\n', dRMSE);
fprintf('  MAE          : %.6f\n', dMAE);
fprintf('  Correlation  : %.6f\n', dCor);
fprintf('  Hit rate     : %.6f\n', dHit);
fprintf('  MZ R²        : %.6f\n', dR2_MZ);
fprintf('  MZ F / p     : %.4f / %.4f\n', dF_MZ, dP_MZ);
fprintf('  OOS period   : %s  to  %s\n', char(dtOOSBeg,'yyyy-MM-dd'), char(dtOOSEnd,'yyyy-MM-dd'));

% =========================================================================
%  6. Save results
% =========================================================================
if ~exist(sOutputDir, 'dir')
    mkdir(sOutputDir);
end

sTimestamp = datestr(now, 'yyyymmdd_HHMMSS');

% --- (a) Step log --------------------------------------------------------
if ~isempty(cStepLog)
    tLog = cell2table(cStepLog, 'VariableNames', {'Step','Action','Predictor','Metric'});
    sLogFile = fullfile(sOutputDir, ['fb_steplog_', sTimestamp, '.csv']);
    writetable(tLog, sLogFile);
    fprintf('\nStep log saved to: %s\n', sLogFile);
end

% --- (b) Final model summary ---------------------------------------------
sFinalPreds = strjoin(cPredNames(vSelected), '; ');
if lRolling; sWindowType = 'rolling'; else; sWindowType = 'expanding'; end
cSummary = { ...
    'Metric',           sMetric; ...
    'WindowType',       sWindowType; ...
    'TrainObs',         num2str(iTrainObs); ...
    'TimeLag',          num2str(iTimeLag); ...
    'NumSelected',      num2str(length(vSelected)); ...
    'SelectedPredictors', sFinalPreds; ...
    'OOS_Beg',          char(dtOOSBeg, 'yyyy-MM-dd'); ...
    'OOS_End',          char(dtOOSEnd, 'yyyy-MM-dd'); ...
    'Num_OOS_Obs',      num2str(sum(lValid)); ...
    'R2_OOS',           num2str(rOOS.vR2OOS,  '%.6f'); ...
    'R2_OOS_CT',        num2str(rOOS.vR2OOSCT,'%.6f'); ...
    'CW_stat',          num2str(rOOS.vCW,     '%.4f'); ...
    'CW_p',             num2str(rOOS.vCWp,    '%.4f'); ...
    'CW_stat_CT',       num2str(rOOS.vCW_CT,  '%.4f'); ...
    'CW_p_CT',          num2str(rOOS.vCWp_CT, '%.4f'); ...
    'DM_stat',          num2str(rOOS.vDM,     '%.4f'); ...
    'DM_p',             num2str(rOOS.vDMp,    '%.4f'); ...
    'RMSE',             num2str(dRMSE,  '%.6f'); ...
    'MAE',              num2str(dMAE,   '%.6f'); ...
    'Cor',              num2str(dCor,   '%.6f'); ...
    'HitRate',          num2str(dHit,   '%.6f'); ...
    'R2_MZ',            num2str(dR2_MZ, '%.6f'); ...
    'F_MZ',             num2str(dF_MZ,  '%.4f'); ...
    'p_MZ',             num2str(dP_MZ,  '%.4f'); ...
};

tSummary  = cell2table(cSummary, 'VariableNames', {'Key','Value'});
sSumFile  = fullfile(sOutputDir, ['fb_summary_', sTimestamp, '.csv']);
writetable(tSummary, sSumFile);
fprintf('Summary saved to:  %s\n', sSumFile);

% Restore path
path(sOldPath);
