% inflation_prediction_forward_backward_oos.m
% Forward-backward stepwise predictor selection performed GENUINELY OUT OF
% SAMPLE: the selection is redone at every rolling-window origin using only
% data available at that point, so no future information leaks into the
% choice of predictors.
%
% At each origin t the in-sample window is split chronologically into a
% real-training part (first 1-dValFrac) and a validation part (last
% dValFrac). The forward-backward search fits each candidate predictor set
% on the training part and scores it on the validation part. The set with
% the best validation metric is then re-fitted on the FULL window and used
% to forecast the next (out-of-sample) observation t+1.
%
% The predictors selected at every origin are logged, and a frequency table
% reports how often each predictor was chosen.
%
% CSV format: col 1 = date, col 2 = target, col 3..end = predictors.

clear; clc; close all;

% =========================================================================
%  CONFIG — edit here
% =========================================================================
sCSVPath    = './DATA/Liedtke/US/aggregated.csv';  % input CSV
sOutputDir  = './RESULTS/';                     % output directory

lRolling    = true;     % true = rolling window, false = expanding window
iTrainObs   = 120;       % in-sample window length (number of observations)
iTimeLag    = 1;        % predictor lag (periods; 1 = standard predictive regression)
dValFrac    = 0.5;     % fraction of each training window used for validation

% Selection metric — choose ONE of:
%   'R2OOS'    OOS R² vs. historical-mean benchmark  (higher is better)
%   'R2OOS_CT' OOS R² with Campbell-Thompson adj.    (higher is better)
%   'RMSE'     Root mean squared error               (lower  is better)
%   'MAE'      Mean absolute error                   (lower  is better)
%   'Cor'      Pearson correlation                   (higher is better)
%   'HitRate'  Directional accuracy                  (higher is better)
%   'AIC'      Akaike info. criterion, in-sample     (lower  is better)
%   'BIC'      Schwarz info. criterion, in-sample    (lower  is better)
%
% Note: AIC/BIC are in-sample criteria that penalise model complexity
% directly. When one of them is chosen the candidate set is fit on the FULL
% rolling window (no train/validation split is used).
sMetric = 'RMSE';
% =========================================================================

% Map metric → direction
cHigherBetter   = {'R2OOS','R2OOS_CT','Cor','HitRate'};
lHigherIsBetter = ismember(sMetric, cHigherBetter);

% Information criteria are scored in-sample on the full window (no split)
lInfoCrit = ismember(sMetric, {'AIC','BIC'});

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
if lInfoCrit
    fprintf('Selection metric: %s  (%s is better, scored in-sample on full window)\n\n', sMetric, sDir);
else
    fprintf('Selection metric: %s  (%s is better, scored on validation)\n\n', sMetric, sDir);
end

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
% ALL predictors have data, so every candidate predictor set is evaluated on
% the exact same observations.
lComplete = ~isnan(vY_raw) & ~any(isnan(mXlag), 2);
vY        = vY_raw(lComplete);
mXlag     = mXlag(lComplete, :);
dtDates   = dtDates_raw(lComplete);
iNumObs   = length(vY);

fprintf('Common complete sample: %d of %d observations retained.\n', ...
    iNumObs, iNumObs_raw);

iNumVal = max(1, round(iTrainObs * dValFrac));
iNumTr  = iTrainObs - iNumVal;
if iNumTr < 2
    error('Training window too small after validation split (train=%d, val=%d).', ...
        iNumTr, iNumVal);
end
if lInfoCrit
    fprintf('Per-origin fit: full window of %d observations (%s, no split).\n\n', ...
        iTrainObs, sMetric);
else
    fprintf('Per-origin split: %d training + %d validation observations.\n\n', ...
        iNumTr, iNumVal);
end

% =========================================================================
%  3. Rolling loop with per-origin forward-backward selection
% =========================================================================
vYhat     = NaN(iNumObs, 1);    % OOS forecast from the selected model
vYhatBM   = NaN(iNumObs, 1);    % benchmark: historical mean
vNumSel   = NaN(iNumObs, 1);    % number of predictors selected at each origin
cSelNames = repmat({''}, iNumObs, 1);  % selected predictor names per origin

vSelCount = zeros(iNumPred, 1); % how often each predictor was selected
iNumOrig  = 0;                  % number of forecast origins produced

for iIdxT = iTrainObs : iNumObs - 1
    % In-sample window
    if lRolling
        vIn = (iIdxT - iTrainObs + 1) : iIdxT;
    else
        vIn = 1 : iIdxT;
    end
    iOut = iIdxT + 1;            % one step ahead (out of sample)

    if any(isnan(vY(vIn))) || isnan(vY(iOut))
        continue
    end

    % Decide fit / scoring rows: in-sample criteria use the full window,
    % holdout metrics fit on the first part and score on the validation tail.
    if lInfoCrit
        vIdxFit   = vIn;
        vIdxScore = vIn;
    else
        iNv       = max(1, round(numel(vIn) * dValFrac));
        vIdxFit   = vIn(1 : end - iNv);
        vIdxScore = vIn(end - iNv + 1 : end);
    end

    % --- Forward-backward selection -------------------------------------
    vSel = fSelectFB(vY, mXlag, iNumPred, vIdxFit, vIdxScore, sMetric, lHigherIsBetter);

    % --- Re-fit selected set on the FULL window, forecast t+1 -----------
    vYhat(iOut)   = fPredictSet(vY, mXlag, vSel, vIn, iOut);
    vYhatBM(iOut) = mean(vY(vIn));

    % --- Record selection ----------------------------------------------
    vNumSel(iOut)   = numel(vSel);
    if isempty(vSel)
        cSelNames{iOut} = '(none)';
    else
        cSelNames{iOut} = strjoin(cPredNames(vSel), '; ');
        vSelCount(vSel) = vSelCount(vSel) + 1;
    end
    iNumOrig = iNumOrig + 1;
end

fprintf('Produced %d out-of-sample forecasts.\n\n', iNumOrig);

% =========================================================================
%  4. Metrics on the OOS forecasts
% =========================================================================
lValid = ~isnan(vY) & ~isnan(vYhat);

rOOS = fEvaluatePerformanceOOS(vY, vYhatBM, vYhat);

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
    dtOOSBeg = NaT; dtOOSEnd = NaT;
else
    dtOOSBeg = dtDates(iFirst);
    dtOOSEnd = dtDates(iLast);
end

fprintf('========== Final OOS metrics ===========\n');
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
if ~isempty(iFirst)
    fprintf('  OOS period   : %s  to  %s\n', ...
        char(dtOOSBeg,'yyyy-MM-dd'), char(dtOOSEnd,'yyyy-MM-dd'));
end
fprintf('  Avg # selected predictors / origin: %.2f\n\n', mean(vNumSel,'omitnan'));

% =========================================================================
%  5. Predictor-selection frequency
% =========================================================================
if iNumOrig > 0
    vFrac = vSelCount / iNumOrig;
else
    vFrac = zeros(iNumPred,1);
end
[~, vOrd] = sort(vSelCount, 'descend');

tFreq = table(cPredNames(vOrd)', vSelCount(vOrd), vFrac(vOrd), ...
    'VariableNames', {'Predictor','TimesSelected','FracOrigins'});

fprintf('========== Predictor selection frequency ===========\n');
disp(tFreq);

% =========================================================================
%  6. Save results
% =========================================================================
if ~exist(sOutputDir, 'dir')
    mkdir(sOutputDir);
end
sTimestamp = datestr(now, 'yyyymmdd_HHMMSS');

% --- (a) Per-origin predictions + selected predictors --------------------
tPred = table(dtDates, vY, vYhat, vYhatBM, vNumSel, cSelNames, ...
    'VariableNames', {'Date','Actual','Forecast','Benchmark','NumSelected','SelectedPredictors'});
sPredFile = fullfile(sOutputDir, ['fb_oos_predictions_', sTimestamp, '.csv']);
writetable(tPred, sPredFile);
fprintf('\nPredictions saved to:        %s\n', sPredFile);

% --- (b) Selection-frequency table ---------------------------------------
sFreqFile = fullfile(sOutputDir, ['fb_oos_selection_freq_', sTimestamp, '.csv']);
writetable(tFreq, sFreqFile);
fprintf('Selection frequency saved to: %s\n', sFreqFile);

% --- (c) Summary ---------------------------------------------------------
if lRolling; sWindowType = 'rolling'; else; sWindowType = 'expanding'; end
cSummary = { ...
    'Metric',           sMetric; ...
    'WindowType',       sWindowType; ...
    'TrainObs',         num2str(iTrainObs); ...
    'ValFrac',          num2str(dValFrac); ...
    'TimeLag',          num2str(iTimeLag); ...
    'Num_OOS_Obs',      num2str(sum(lValid)); ...
    'Avg_NumSelected',  num2str(mean(vNumSel,'omitnan'), '%.4f'); ...
    'OOS_Beg',          char(dtOOSBeg, 'yyyy-MM-dd'); ...
    'OOS_End',          char(dtOOSEnd, 'yyyy-MM-dd'); ...
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
tSummary = cell2table(cSummary, 'VariableNames', {'Key','Value'});
sSumFile = fullfile(sOutputDir, ['fb_oos_summary_', sTimestamp, '.csv']);
writetable(tSummary, sSumFile);
fprintf('Summary saved to:            %s\n', sSumFile);

% Restore path
path(sOldPath);

% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================
function vSel = fSelectFB(vY_, mXlag_, iNumPred_, vIdxFit, vIdxScore, sMetric_, lHB_)
% Forward-backward selection scored by fValMetric.
    vSel = [];
    dCur = fValMetric(vY_, mXlag_, vSel, vIdxFit, vIdxScore, sMetric_, lHB_);

    lChanged = true;
    while lChanged
        lChanged = false;

        % --- Forward: add the predictor that most improves the score ----
        vCand    = setdiff(1:iNumPred_, vSel);
        dBestAdd = dCur; iAdd = -1;
        for p = vCand
            vTry = sort([vSel, p]);
            d    = fValMetric(vY_, mXlag_, vTry, vIdxFit, vIdxScore, sMetric_, lHB_);
            if (lHB_ && d > dBestAdd) || (~lHB_ && d < dBestAdd)
                dBestAdd = d; iAdd = p;
            end
        end
        if iAdd > 0
            vSel = sort([vSel, iAdd]);
            dCur = dBestAdd;
            lChanged = true;
        end

        % --- Backward: remove any predictor that improves the score -----
        lRem = true;
        while lRem && ~isempty(vSel)
            lRem     = false;
            dBestRem = dCur; iRem = -1;
            for j = 1:numel(vSel)
                vTry = vSel; vTry(j) = [];
                d    = fValMetric(vY_, mXlag_, vTry, vIdxFit, vIdxScore, sMetric_, lHB_);
                if (lHB_ && d > dBestRem) || (~lHB_ && d < dBestRem)
                    dBestRem = d; iRem = j;
                end
            end
            if iRem > 0
                vSel(iRem) = [];
                dCur = dBestRem;
                lChanged = true;
                lRem = true;
            end
        end
    end
end

function d = fValMetric(vY_, mXlag_, vIdxSel, vIdxFit, vIdxScore, sMetric_, lHB_)
% Fit a predictor set on vIdxFit rows, score it on vIdxScore rows.
% Holdout metrics use a validation tail; AIC/BIC use the (in-sample) full
% window where vIdxScore == vIdxFit and the criterion penalises complexity.
    if numel(vIdxScore) < 3
        if lHB_; d = -Inf; else; d = Inf; end
        return
    end
    vHat  = fPredictSet(vY_, mXlag_, vIdxSel, vIdxFit, vIdxScore);
    vTrue = vY_(vIdxScore);

    switch sMetric_
        case {'R2OOS','R2OOS_CT'}
            vBM  = repmat(mean(vY_(vIdxFit)), numel(vIdxScore), 1);
            rOOS = fEvaluatePerformanceOOS(vTrue, vBM, vHat);
            if strcmp(sMetric_, 'R2OOS')
                d = rOOS.vR2OOS;
            else
                d = rOOS.vR2OOSCT;
            end
        case {'RMSE','MAE','Cor','HitRate'}
            rQ = fForecastQuality(vTrue, vHat);
            switch sMetric_
                case 'RMSE';    d = rQ.vRMSE;
                case 'MAE';     d = rQ.vMAE;
                case 'Cor';     d = rQ.vCor;
                case 'HitRate'; d = rQ.vHitRate;
            end
        case {'AIC','BIC'}
            % In-sample residual variance (MLE) and parameter count
            vRes  = vTrue - vHat;
            iN    = numel(vRes);
            iK    = numel(vIdxSel) + 1;           % +1 for the intercept
            dSig2 = sum(vRes.^2) / iN;
            if dSig2 <= 0
                d = -Inf;                         % perfect fit -> lowest IC
                return
            end
            if strcmp(sMetric_, 'AIC')
                d = iN*log(dSig2) + 2*iK;
            else
                d = iN*log(dSig2) + iK*log(iN);
            end
        otherwise
            error('Unknown metric: %s', sMetric_);
    end
    if isnan(d)
        if lHB_; d = -Inf; else; d = Inf; end
    end
end

function vYp = fPredictSet(vY_, mXlag_, vIdxSel, vIdxFit, vIdxPred)
% OLS fit of a predictor set on vIdxFit rows, prediction for vIdxPred rows.
% Empty set -> historical mean of the fit window.
    if isempty(vIdxSel)
        vYp = repmat(mean(vY_(vIdxFit)), numel(vIdxPred), 1);
        return
    end
    mXin  = [ones(numel(vIdxFit), 1),  mXlag_(vIdxFit,  vIdxSel)];
    vBeta = mXin \ vY_(vIdxFit);
    mXout = [ones(numel(vIdxPred), 1), mXlag_(vIdxPred, vIdxSel)];
    vYp   = mXout * vBeta;
end
