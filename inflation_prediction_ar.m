% inflation_prediction_ar.m
% Pure autoregressive (one-step-ahead) OOS inflation forecast using the VAR
% utilities from bKap4 (fEstVAR / fPredictVAR).
%
% The model predicts the target from its OWN past values only. A reporting
% lag accounts for publication delay: when standing at the target date t,
% the most recent value you may use is y(t-1-iReportLag).
%
%   Regressors for target y(t):  y(t-(r+1)), y(t-(r+2)), ..., y(t-(r+p))
%       r = iReportLag   (reporting/publication delay)
%       p = lookback     (number of AR lags)
%
%   iReportLag = 0  ->  uses t-1, t-2, ...
%   iReportLag = 1  ->  uses t-2, t-3, ...
%
% Optional: select the lookback p automatically by AIC (same idea as
% fEstOptVAR, adapted for the reporting-lag offset).
%
%   CSV format (1 data column + date):
%     col 1 = date
%     col 2 = target (the series to predict)

clear; clc; close all;

% =========================================================================
%  CONFIG — edit here
% =========================================================================
sCSVPath    = './DATA/Liedtke/US/aggregated.csv';    % input CSV (date, target)
sOutputDir  = './RESULTS/';                     % output directory

lRolling      = true;     % true = rolling window, false = expanding window
iTrainObs     = 60;       % in-sample window length (number of observations)
iReportLag    = 1;        % reporting lag r: first usable lag is y(t-(r+1))
iLookback     = 1;        % lookback p: number of AR lags (used if not optimal)

lOptimalLookback = true; % true = pick p by AIC at each training window
vLookbackGrid    = 1:12;   % candidate lookback values when lOptimalLookback
% =========================================================================

% Add utilities
sOldPath = path;
addpath('./Utils/');

% =========================================================================
%  1. Load data
% =========================================================================
tData = readtable(sCSVPath);

dtDates = tData{:,1};
if ~isdatetime(dtDates)
    dtDates = datetime(dtDates);
end

vY      = tData{:,2};     % target to be predicted
iNumObs = length(vY);

fprintf('Loaded %d observations from %s\n', iNumObs, sCSVPath);
fprintf('Reporting lag r = %d  ->  first usable lag is y(t-%d)\n', ...
    iReportLag, iReportLag + 1);
if lOptimalLookback
    fprintf('Lookback: optimal by AIC over [%s]\n\n', num2str(vLookbackGrid));
else
    fprintf('Lookback p = %d\n\n', iLookback);
end

% =========================================================================
%  2. Build the full lagged-regressor matrix
%     mRegMax(:,k) = y(t-(r+k))  for k = 1..pMax
%     A model with lookback p simply uses the first p columns.
% =========================================================================
if lOptimalLookback
    iPmax = max(vLookbackGrid);
else
    iPmax = iLookback;
end
mRegMax = lagmatrix(vY, (iReportLag + 1) : (iReportLag + iPmax));

% =========================================================================
%  3. Rolling/expanding one-step-ahead loop
% =========================================================================
vYhat     = NaN(iNumObs, 1);    % AR forecast
vYhatBM   = NaN(iNumObs, 1);    % benchmark: historical mean
vLagUsed  = NaN(iNumObs, 1);    % lookback actually used at each step

for iIdxT = iTrainObs : iNumObs - 1
    % In-sample index
    if lRolling
        vIdxIn = (iIdxT - iTrainObs + 1) : iIdxT;
    else
        vIdxIn = 1 : iIdxT;
    end
    iIdxOut = iIdxT + 1;            % one step ahead

    vYin = vY(vIdxIn);
    if any(isnan(vYin))
        continue
    end

    % --- Choose lookback p ---------------------------------------------
    if lOptimalLookback
        % AIC-based selection over the candidate grid (cf. fEstOptVAR)
        dBestAIC = Inf; iBestP = NaN; rBest = [];
        for p = vLookbackGrid(:)'
            mXin_p  = mRegMax(vIdxIn,  1:p);
            mXout_p = mRegMax(iIdxOut, 1:p);
            if any(isnan(mXin_p(:))) || any(isnan(mXout_p))
                continue
            end
            rTry = fEstVAR(vYin, mXin_p, 'iNumLags', 0, 'lEstAlpha', true);
            if rTry.dAIC < dBestAIC
                dBestAIC = rTry.dAIC; iBestP = p; rBest = rTry;
            end
        end
        if isempty(rBest)
            continue        % no candidate had complete data
        end
        iP     = iBestP;
        rModel = rBest;
        mXout  = mRegMax(iIdxOut, 1:iP);
    else
        iP      = iLookback;
        mXin    = mRegMax(vIdxIn,  1:iP);
        mXout   = mRegMax(iIdxOut, 1:iP);
        if any(isnan(mXin(:))) || any(isnan(mXout))
            continue
        end
        rModel  = fEstVAR(vYin, mXin, 'iNumLags', 0, 'lEstAlpha', true);
    end

    % --- One-step-ahead prediction -------------------------------------
    mYhat_step       = fPredictVAR(rModel, 1, mXout);
    vYhat(iIdxOut)   = mYhat_step(1);
    vYhatBM(iIdxOut) = mean(vYin);     % historical-mean benchmark
    vLagUsed(iIdxOut)= iP;
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

fprintf('========== AR model metrics ===========\n');
fprintf('  OOS obs      : %d\n', sum(lValid));
if lOptimalLookback
    fprintf('  Lookback used: %d to %d (median %g)\n', ...
        min(vLagUsed,[],'omitnan'), max(vLagUsed,[],'omitnan'), ...
        median(vLagUsed,'omitnan'));
end
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
if lOptimalLookback
    sLookback = ['optimal[', num2str(vLookbackGrid), ']'];
else
    sLookback = num2str(iLookback);
end

% --- (a) Summary ---------------------------------------------------------
cSummary = { ...
    'WindowType',   sWindowType; ...
    'TrainObs',     num2str(iTrainObs); ...
    'ReportLag',    num2str(iReportLag); ...
    'Lookback',     sLookback; ...
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
sSumFile = fullfile(sOutputDir, ['ar_summary_', sTimestamp, '.csv']);
writetable(tSummary, sSumFile);
fprintf('\nSummary saved to:     %s\n', sSumFile);

% --- (b) Predictions -----------------------------------------------------
tPred = table(dtDates, vY, vYhat, vYhatBM, vLagUsed, ...
    'VariableNames', {'Date','Actual','Forecast','Benchmark','LookbackUsed'});
sPredFile = fullfile(sOutputDir, ['ar_predictions_', sTimestamp, '.csv']);
writetable(tPred, sPredFile);
fprintf('Predictions saved to: %s\n', sPredFile);

% Restore path
path(sOldPath);
