% inflation.m — Rolling-window inflation forecasting
% Models: (a) Linear Regression, (b) AR, (c) RNN
%
% CSV format:  col 1 = date (any parseable string), col 2 = target (inflation),
%              remaining cols = additional predictors

clear; clc; close all;
rng(42);

% =========================================================================
%  OPTIONS — edit these
% =========================================================================
sCSVPath        = './DATA/inflation_data.csv';  % path to CSV file
iWindowSize     = 60;       % rolling in-sample window length (observations)
lRolling        = true;     % true = rolling window, false = expanding window
iStepSize       = 1;        % out-of-sample step size (observations)
iHorizon        = 1;        % prediction horizon (cumprod over this many steps)
iARLags         = 1;        % number of AR lags
iRNNUnits       = 3;        % number of hidden units in RNN
iRNNMaxIter     = 500;      % max optimisation iterations for RNN
% =========================================================================

% Add utility paths
sOldPath = path;
addpath('./Utils/');
addpath(genpath('./RNN'));

% =========================================================================
%  1. Load data
% =========================================================================
tData   = readtable(sCSVPath);
dtDates = tData{:,1};
if ~isdatetime(dtDates)
    dtDates = datetime(dtDates);
end
mData   = tData{:,2:end};          % col 1 = target, rest = predictors
vY_raw  = mData(:,1);              % target (e.g. monthly inflation)
mX_raw  = mData(:,2:end);          % predictors (may be empty)

% =========================================================================
%  2. Build compound target over the forecast horizon
%     If iHorizon == 1 the target is unchanged (simple return / rate).
%     If iHorizon > 1 we compute cumprod of (1 + r) over the horizon and
%     subtract 1, then shift so that vY(t) is the h-period return starting
%     at t+1.
% =========================================================================
iNumObs = length(vY_raw);

if iHorizon > 1
    vY_comp = NaN(iNumObs,1);
    for iIdxT = 1 : iNumObs - iHorizon
        vY_comp(iIdxT) = prod(1 + vY_raw(iIdxT+1 : iIdxT+iHorizon)) - 1;
    end
    vY = vY_comp;
else
    vY = vY_raw;
end

% Lag predictors by 1 so that mX(t) predicts vY(t)  (predictive regression)
if ~isempty(mX_raw)
    mXlag = [NaN(1, size(mX_raw,2)); mX_raw(1:end-1,:)];
else
    mXlag = zeros(iNumObs, 0);   % empty — AR / const only
end

% Remove rows with any NaN in target or predictors
lIsNaN  = isnan(vY) | any(isnan(mXlag),2);
vY      = vY(~lIsNaN);
mXlag   = mXlag(~lIsNaN,:);
dtDates = dtDates(~lIsNaN);
iNumObs = length(vY);

fprintf('Observations after cleaning: %d\n', iNumObs);
assert(iNumObs > iWindowSize, ...
    'Not enough observations for the chosen window size.');

% =========================================================================
%  3. Rolling-window loop
% =========================================================================
vYhat_LR  = NaN(iNumObs,1);
vYhat_AR  = NaN(iNumObs,1);
vYhat_RNN = NaN(iNumObs,1);

% RNN model struct
rRNN.iNumUnits                  = iRNNUnits;
rRNN.rHidLayer.lEstBias         = true;
rRNN.rHidLayer.iActivFun        = 3;      % tanh
rRNN.rOutLayer.lEstAlpha        = false;
rRNN.rOutLayer.iActivFun        = 1;      % linear
rRNN.rErrCorrLayer.iMode        = 1;
rRNN.rErrCorrLayer.iNumUnits    = 0;
rRNN.rErrCorrLayer.iActivFun    = 3;
rRNN.rOptimOpt.iObjFun          = 1;      % SSR
rRNN.rOptimOpt.sSolver          = 'rprob';
rRNN.rOptimOpt.iMaxIter         = iRNNMaxIter;
rRNN.rOptimOpt.sDisplay         = 'off';
rRNN.rOptimOpt.iMaxTime         = Inf;
rRNN.rOptimOpt.vLB              = -100;
rRNN.rOptimOpt.vUB              = 100;

iNumPreds = size(mXlag, 2);

iIdxT = iWindowSize;
while iIdxT <= iNumObs - iStepSize
    % --- in-sample index ---
    if lRolling
        vIdxIn = (iIdxT - iWindowSize + 1) : iIdxT;
    else
        vIdxIn = 1 : iIdxT;
    end

    % --- out-of-sample indices (up to step size) ---
    vIdxOut = (iIdxT + 1) : min(iIdxT + iStepSize, iNumObs);

    vYin  = vY(vIdxIn);
    mXin  = mXlag(vIdxIn,:);
    vYout = vY(vIdxOut);
    mXout = mXlag(vIdxOut,:);

    % --- (a) Linear Regression ---
    mXin_LR  = [ones(length(vIdxIn),1), mXin];
    mXout_LR = [ones(length(vIdxOut),1), mXout];
    vBeta    = mXin_LR \ vYin;
    vYhat_LR(vIdxOut) = mXout_LR * vBeta;

    % --- (b) AR model via fEstVAR (univariate = AR) ---
    % Build predictor matrix: AR lags of vY only (mX used as exogenous if available)
    if iNumPreds > 0
        rModelAR = fEstVAR(vYin, mXin, 'iNumLags', iARLags, 'lEstAlpha', true);
    else
        rModelAR = fEstVAR(vYin, [], 'iNumLags', iARLags, 'lEstAlpha', true);
    end
    % One-step-ahead prediction per out-of-sample point
    for iIdxOOS = 1:length(vIdxOut)
        if iNumPreds > 0
            mYhat_step = fPredictVAR(rModelAR, 1, mXout(iIdxOOS,:));
        else
            mYhat_step = fPredictVAR(rModelAR, 1, []);
        end
        vYhat_AR(vIdxOut(iIdxOOS)) = mYhat_step(1);
        % Update model's last observation for multi-step (append realisation)
        rModelAR.mY = [rModelAR.mY; vYin(end)];
    end

    % --- (c) RNN ---
    % RNN expects (N x T) matrices — transpose
    mYin_RNN  = vYin';            % 1 x T_in
    if iNumPreds > 0
        mXin_RNN  = mXin';        % K x T_in
        mXout_RNN = mXout';       % K x T_out
    else
        % Feed a constant predictor so the RNN has at least one input
        mXin_RNN  = ones(1, length(vIdxIn));
        mXout_RNN = ones(1, length(vIdxOut));
    end
    mYout_RNN = vYout';

    try
        rRNNtrained = fEstRNN(rRNN, mYin_RNN, mXin_RNN, []);
        mYhat_step  = fPredictRNN(rRNNtrained, mYout_RNN, mXout_RNN);
        vYhat_RNN(vIdxOut) = mYhat_step';
    catch ME
        warning('RNN failed at t=%d: %s', iIdxT, ME.message);
    end

    iIdxT = iIdxT + iStepSize;
end

% =========================================================================
%  4. Statistics: t, p, R, R2 for each model
% =========================================================================
% Use realized vs. predicted; regress realized on predicted to get
% slope t / p, and compute Pearson r and R2.

cModelNames = {'LinReg','AR','RNN'};
mYhat_all   = [vYhat_LR, vYhat_AR, vYhat_RNN];

fprintf('\n========== Forecast Quality ===========\n');
for iIdxM = 1:3
    vPred = mYhat_all(:,iIdxM);
    lValid = ~isnan(vY) & ~isnan(vPred);
    vYv   = vY(lValid);
    vPv   = vPred(lValid);
    iN    = sum(lValid);

    if iN < 3
        fprintf('%s: insufficient valid predictions\n', cModelNames{iIdxM});
        continue;
    end

    % R and R2 (Pearson)
    mCorr   = corrcoef(vYv, vPv);
    dR      = mCorr(1,2);
    dR2     = dR^2;

    % Mincer-Zarnowitz regression: y = a + b*yhat + e
    mXreg   = [ones(iN,1), vPv];
    [vB, ~, ~, ~, stats] = regress(vYv, mXreg);
    dR2_reg = stats(1);
    dF      = stats(2);
    dP      = stats(3);

    % t-statistic on slope coefficient
    vResid  = vYv - mXreg * vB;
    dS2     = sum(vResid.^2) / (iN - 2);
    mXXinv  = inv(mXreg' * mXreg);
    dSE_b1  = sqrt(dS2 * mXXinv(2,2));
    dT      = vB(2) / dSE_b1;
    dP_t    = 2 * (1 - tcdf(abs(dT), iN - 2));

    fprintf('\n--- %s ---\n', cModelNames{iIdxM});
    fprintf('  Valid predictions : %d\n', iN);
    fprintf('  Pearson r         : %.4f\n', dR);
    fprintf('  R2 (Pearson)      : %.4f\n', dR2);
    fprintf('  MZ R2             : %.4f\n', dR2_reg);
    fprintf('  MZ F-stat         : %.4f\n', dF);
    fprintf('  MZ p-value        : %.4f\n', dP);
    fprintf('  Slope t-stat      : %.4f\n', dT);
    fprintf('  Slope p-value     : %.4f\n', dP_t);
    fprintf('  Intercept         : %.4f\n', vB(1));
    fprintf('  Slope             : %.4f\n', vB(2));
end

% Also use fForecastQuality from Utils
fprintf('\n========== fForecastQuality Results ===========\n');
for iIdxM = 1:3
    vPred = mYhat_all(:,iIdxM);
    lValid = ~isnan(vY) & ~isnan(vPred);
    if sum(lValid) < 3; continue; end
    [rQ, tT] = fForecastQuality(vY(lValid), vPred(lValid));
    fprintf('\n%s — RMSE: %.4f | MAE: %.4f | Cor: %.4f | R2: %.4f | F: %.4f | p: %.4f\n', ...
        cModelNames{iIdxM}, rQ.vRMSE, rQ.vMAE, rQ.vCor, ...
        rQ.rRegResults.vR2, rQ.rRegResults.vF, rQ.rRegResults.vP);
end

% =========================================================================
%  5. Plots
% =========================================================================
lPlot = ~isnan(vY) & (any(~isnan(mYhat_all),2));

figure('Name','Inflation Forecasts','Position',[100 100 1200 800]);
cColors = {'#0072BD','#D95319','#77AC30'};

for iIdxM = 1:3
    subplot(3,1,iIdxM);
    vPred   = mYhat_all(:,iIdxM);
    lValid  = ~isnan(vY) & ~isnan(vPred);

    plot(dtDates, vY, 'k-', 'LineWidth',1.2, 'DisplayName','Realized'); hold on;
    plot(dtDates(lValid), vPred(lValid), '-', ...
        'Color', cColors{iIdxM}, 'LineWidth',1.2, ...
        'DisplayName', cModelNames{iIdxM});

    % Compute stats for annotation
    if sum(lValid) >= 3
        mCorr   = corrcoef(vY(lValid), vPred(lValid));
        dR_ann  = mCorr(1,2);
        [~,~,~,~,stats] = regress(vY(lValid), ...
            [ones(sum(lValid),1), vPred(lValid)]);
        title(sprintf('%s  |  R = %.3f  |  R² = %.3f  |  p = %.4f', ...
            cModelNames{iIdxM}, dR_ann, dR_ann^2, stats(3)));
    else
        title(cModelNames{iIdxM});
    end

    ylabel('Inflation');
    legend('Location','best');
    grid on;
    if iIdxM < 3
        set(gca,'XTickLabel',[]);
    else
        xlabel('Date');
    end
end

sgtitle(sprintf('Rolling-Window Inflation Forecasts  (window=%d, step=%d, horizon=%d)', ...
    iWindowSize, iStepSize, iHorizon));

% =========================================================================
%  6. Residual / error plot
% =========================================================================
figure('Name','Forecast Errors','Position',[100 100 1200 500]);
for iIdxM = 1:3
    vPred  = mYhat_all(:,iIdxM);
    lValid = ~isnan(vY) & ~isnan(vPred);
    vErr   = vY - vPred;
    subplot(1,3,iIdxM);
    plot(dtDates(lValid), vErr(lValid), 'Color', cColors{iIdxM});
    yline(0,'k--');
    title(sprintf('%s Errors', cModelNames{iIdxM}));
    xlabel('Date'); ylabel('Error');
    grid on;
end
sgtitle('Forecast Errors (Realized − Predicted)');

% Restore path
path(sOldPath);
