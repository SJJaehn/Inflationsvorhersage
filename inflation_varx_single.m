% inflation_varx_single.m
% VARX one-step-ahead OOS inflation forecast for EACH predictor individually:
%
%   y(t) = alpha + sum_k phi_k * y(t-(r+1)) + theta * x(t) + e(t)
%
% Mirrors bKap4_3_DemoVectorAutoregressiveModelX_OutOfSample but loops over
% predictors one by one. For each predictor two models are compared:
%   AR   : inflation ~ AR lags only  (reporting-lag adjusted)
%   VARX : AR lags + that one predictor
%
% SHARED_TIMEFRAME = false (default): each predictor uses its own available
%   sample (rows where y + AR lags + that predictor are complete).
% SHARED_TIMEFRAME = true: all predictors are evaluated on the same window
%   (rows where y + AR lags + ALL predictors are complete), matching the
%   behaviour of bKap4_3 exactly.

clear; clc; close all;

sOldPath = path;
addpath('./Utils/');

% =========================================================================
%  Settings
% =========================================================================
sCountry         = fCfg('COUNTRY',         'US');
sDataPath        = ['./DATA/Liedtke/', sCountry, '/'];
sOutputDir       = './RESULTS/';

iReportLag       = 1;
iNumLags         = fCfg('NUM_LAGS',        1);
iNumIn           = fCfg('MIN_INSAMPLE',    120);
lRoll            = fCfg('ROLLING',         true);
lSharedTimeframe = fCfg('SHARED_TIMEFRAME', false);

% =========================================================================
%  Load data
% =========================================================================
tData   = readtable([sDataPath, 'aggregated.csv']);
dtDates = tData{:,1};
if ~isdatetime(dtDates); dtDates = datetime(dtDates); end

vY      = tData{:,2};
mX_raw  = tData{:,3:end};
cXnames = tData.Properties.VariableNames(3:end);
iNumPred = size(mX_raw, 2);

fprintf('Loaded %d observations, %d predictors\n', numel(vY), iNumPred);

% Predictive lag on exogenous predictors (matches Python util.apply_lag(X,1))
iTimeLag = 1;
mX_lag   = [NaN(iTimeLag, iNumPred); mX_raw(1:end-iTimeLag, :)];

% AR lag matrix: y(t-(r+1)) .. y(t-(r+p))
mYlag = lagmatrix(vY, (iReportLag+iTimeLag):(iReportLag+iTimeLag+iNumLags-1));

% =========================================================================
%  Collinearity detection (same as bKap4_3)
% =========================================================================
lExcl = false(1, iNumPred);
for jX = 1:iNumPred
    for kL = 1:size(mYlag, 2)
        vOK = ~isnan(mX_lag(:,jX)) & ~isnan(mYlag(:,kL));
        if nnz(vOK) > 2
            vA = mX_lag(vOK,jX)  - mean(mX_lag(vOK,jX));
            vB = mYlag(vOK,kL)   - mean(mYlag(vOK,kL));
            dRho = (vA'*vB) / sqrt((vA'*vA)*(vB'*vB));
            if abs(dRho) >= 1 - 1e-8; lExcl(jX) = true; break; end
        end
    end
end
if any(lExcl)
    fprintf('Skipped (collinear with AR lag): %s\n', ...
        strjoin(cXnames(lExcl), ', '));
end

% =========================================================================
%  Shared timeframe mask (used when lSharedTimeframe = true)
% =========================================================================
if lSharedTimeframe
    lShared = ~isnan(vY) & ~any(isnan(mYlag), 2) & ~any(isnan(mX_lag), 2);
    fprintf('Shared timeframe: %d of %d observations retained.\n', ...
        sum(lShared), numel(vY));
end

% =========================================================================
%  Loop over predictors
% =========================================================================
cResults = {};

for jX = 1:iNumPred
    if lExcl(jX); continue; end
    sPred = cXnames{jX};

    % --- Filter rows for this predictor ---------------------------------
    if lSharedTimeframe
        lOK = lShared;
    else
        lOK = ~isnan(vY) & ~any(isnan(mYlag), 2) & ~isnan(mX_lag(:,jX));
    end

    vYj      = vY(lOK);
    mYlagj   = mYlag(lOK, :);
    vXj      = mX_lag(lOK, jX);
    dtDatesj = dtDates(lOK);
    iNj      = numel(vYj);
    mZj      = [mYlagj, vXj];   % VARX regressors

    fprintf('[%d/%d] %-30s  n=%d ... ', jX, iNumPred, sPred, iNj);

    % --- OOS loop (mirrors bKap4_3) -------------------------------------
    vYhat_ar = NaN(iNj, 1);
    vYhat_vx = NaN(iNj, 1);
    vYhat_bm = NaN(iNj, 1);

    for iIdxT = iNumIn : iNj - 1
        if lRoll
            vIdxIn = (iIdxT - iNumIn + 1) : iIdxT;
        else
            vIdxIn = 1 : iIdxT;
        end
        iIdxOut = iIdxT + 1;

        vYin     = vYj(vIdxIn);
        mYlagIn  = mYlagj(vIdxIn, :);
        vYlagOut = mYlagj(iIdxOut, :);
        mZin     = mZj(vIdxIn, :);
        vZout    = mZj(iIdxOut, :);

        if any(isnan(vYin)) || any(isnan(mZin), 'all') || any(isnan(vZout))
            continue
        end

        vYhat_bm(iIdxOut) = mean(vYin);

        rAR = fEstVAR(vYin, mYlagIn, 'iNumLags', 0, 'lEstAlpha', true);
        vYhat_ar(iIdxOut) = fPredictVAR(rAR, 1, vYlagOut);

        rVX = fEstVAR(vYin, mZin, 'iNumLags', 0, 'lEstAlpha', true);
        vYhat_vx(iIdxOut) = fPredictVAR(rVX, 1, vZout);
    end

    % --- Evaluate -------------------------------------------------------
    rOOS_ar = fEvaluatePerformanceOOS(vYj, vYhat_bm, vYhat_ar);
    rOOS_vx = fEvaluatePerformanceOOS(vYj, vYhat_bm, vYhat_vx);

    lValid_vx = ~isnan(vYj) & ~isnan(vYhat_vx);
    lValid_ar = ~isnan(vYj) & ~isnan(vYhat_ar);

    if sum(lValid_vx) >= 3
        rQ_vx    = fForecastQuality(vYj(lValid_vx), vYhat_vx(lValid_vx));
        dRMSE_vx = rQ_vx.vRMSE; dMAE_vx = rQ_vx.vMAE;
        dCor_vx  = rQ_vx.vCor;  dHit_vx = rQ_vx.vHitRate;
    else
        dRMSE_vx = NaN; dMAE_vx = NaN; dCor_vx = NaN; dHit_vx = NaN;
    end
    if sum(lValid_ar) >= 3
        rQ_ar    = fForecastQuality(vYj(lValid_ar), vYhat_ar(lValid_ar));
        dRMSE_ar = rQ_ar.vRMSE;
    else
        dRMSE_ar = NaN;
    end

    iFirst = find(lValid_vx, 1, 'first');
    iLast  = find(lValid_vx, 1, 'last');
    if isempty(iFirst)
        sBeg = ''; sEnd = '';
    else
        sBeg = char(dtDatesj(iFirst), 'yyyy-MM-dd');
        sEnd = char(dtDatesj(iLast),  'yyyy-MM-dd');
    end

    fprintf('VARX R2=%+.4f  AR R2=%+.4f  RMSE=%.4f\n', ...
        rOOS_vx.vR2OOS, rOOS_ar.vR2OOS, dRMSE_vx);

    cResults(end+1, :) = { ...
        sPred, sBeg, sEnd, sum(lValid_vx), ...
        rOOS_vx.vR2OOS, rOOS_vx.vCW,  rOOS_vx.vCWp, ...
        rOOS_vx.vDM,    rOOS_vx.vDMp, ...
        dRMSE_vx, dMAE_vx, dCor_vx, dHit_vx, ...
        rOOS_ar.vR2OOS, dRMSE_ar, rOOS_ar.vCWp ...
    }; %#ok<AGROW>
end

% =========================================================================
%  Save results
% =========================================================================
if isempty(cResults)
    warning('No valid results to save.');
    path(sOldPath); return
end

cColNames = { ...
    'Predictor','OOS_Beg','OOS_End','Num_OOS_Obs', ...
    'VARX_R2_OOS','VARX_CW_stat','VARX_CW_p','VARX_DM_stat','VARX_DM_p', ...
    'VARX_RMSE','VARX_MAE','VARX_Cor','VARX_HitRate', ...
    'AR_R2_OOS','AR_RMSE','AR_CW_p' ...
};
tOut = cell2table(cResults, 'VariableNames', cColNames);

if lRoll; sWindow = 'rolling'; else; sWindow = 'expanding'; end
sSuffix  = ''; if lSharedTimeframe; sSuffix = '_shared'; end
sOptions = sprintf('min%d_%s_lags%d_report%d%s', ...
    iNumIn, sWindow, iNumLags, iReportLag, sSuffix);
sOutDir  = fResultDir(sOutputDir, 'VAR_single', sCountry, 'oos', sOptions);
sOutFile = fullfile(sOutDir, 'results.csv');
writetable(tOut, sOutFile);

fprintf('\n========== VARX single summary ==========\n');
disp(tOut(:, {'Predictor','VARX_R2_OOS','VARX_CW_p','VARX_RMSE','AR_R2_OOS'}));
fprintf('Results saved to: %s\n', sOutFile);

% =========================================================================
%  AR baseline and Full VARX reference
%    SHARED_TIMEFRAME=false  -> standalone AR results (full long sample)
%    SHARED_TIMEFRAME=true   -> full VAR results (same shared sample)
% =========================================================================
sVarOptions = strrep(sOptions, '_shared', '');
sVarPath = fullfile(sOutputDir, 'VAR', sCountry, 'oos', sVarOptions, 'results.csv');
if lSharedTimeframe
    if isfile(sVarPath)
        tVar = readtable(sVarPath, 'ReadRowNames', true);
        dAR_R2   = str2double(tVar{'AR_R2_OOS','Value'});
        dAR_RMSE = str2double(tVar{'AR_RMSE','Value'});
        dVX_R2   = str2double(tVar{'VARX_R2_OOS','Value'});
        dVX_RMSE = str2double(tVar{'VARX_RMSE','Value'});
        fprintf('\n--- Reference (from full VAR, shared sample) ---\n');
        fprintf('AR baseline  :  R2=%+.4f  RMSE=%.4f\n', dAR_R2, dAR_RMSE);
        fprintf('Full VARX    :  R2=%+.4f  RMSE=%.4f\n', dVX_R2, dVX_RMSE);
    else
        fprintf('(full VAR results not found at %s)\n', sVarPath);
    end
else
    sArOptions = sprintf('train%d_%s_report%d_p%d', iNumIn, sWindow, iReportLag, iNumLags);
    sArPath = fullfile(sOutputDir, 'AR', sCountry, 'oos', sArOptions, 'results.csv');
    if isfile(sArPath)
        tAr = readtable(sArPath, 'ReadRowNames', true);
        dAR_R2   = str2double(tAr{'R2_OOS','Value'});
        dAR_RMSE = str2double(tAr{'RMSE','Value'});
        fprintf('\n--- Reference (standalone AR, full sample) ---\n');
        fprintf('AR baseline  :  R2=%+.4f  RMSE=%.4f\n', dAR_R2, dAR_RMSE);
    else
        fprintf('(AR results not found at %s)\n', sArPath);
    end
    if isfile(sVarPath)
        tVar = readtable(sVarPath, 'ReadRowNames', true);
        dVX_R2   = str2double(tVar{'VARX_R2_OOS','Value'});
        dVX_RMSE = str2double(tVar{'VARX_RMSE','Value'});
        fprintf('Full VARX    :  R2=%+.4f  RMSE=%.4f\n', dVX_R2, dVX_RMSE);
    end
end

path(sOldPath);
