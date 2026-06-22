% Script for demonstrating the implementation of a VAR model

% Clear console
clear; clc; close all;

% Set random number generator
rng(42);

% Set path
sOldPath = path;
addpath('./Utils/');

%% Simulate a bivariate VAR(2) model
% Settings
iNumObs         = 10000;    % Number of time-series observations
iNumInit        = 1000;     % Observations required for initialization

% Simulate Phi matrix
mPhi = cat(3, [0.2, 0.0; -0.3, 0.4],... % First lag
    [-0.1, 0.1; 0.2, -0.3]);

% Determine dimensions
[~,iNumSeries,iNumLags] = size(mPhi);

% Initialize memory
mY                  = randn(iNumObs+iNumInit,iNumSeries) * 0.5; % Random shocks

% Generate data
for iIdxT = (iNumLags+1):(iNumObs+iNumInit)
    % Get the component of y due to random noise
    vYnoise = mY(iIdxT,:)';

    % Get the component of y due to lags
    vYlag   = zeros(iNumSeries, 1);
    for iIdxP = 1:iNumLags
        vYlag = vYlag + mPhi(:,:,iIdxP) * mY(iIdxT-iIdxP,:)';
    end

    % Add together
    mY(iIdxT,:) = vYnoise + vYlag;
end

% Remove initialization period
mY(1:iNumInit,:) = [];

% Check (auto)correlation
[mCorrCoef, mCorrCoefP] = fGetAutoCorrFun(mY(:,1), mY);
bar(mCorrCoef);

%% Estimation
% Estimate VAR(1) model
rModelVAR1 = fEstVAR(mY, [], 'iNumLags', 1, 'lEstAlpha', true);

% Estimate VAR(2) model
rModelVAR2 = fEstVAR(mY, [], 'iNumLags', 2, 'lEstAlpha', true);

% Get difference in estimated coefficients
mDiff = rModelVAR2.mPhi - mPhi;
fprintf('Mean absolute difference in phi coefficients: %.4f \n', mean(abs(mDiff),'all'))

% Select best lag using AIC
rModelOpt = fEstOptVAR(mY, []);
fprintf('Optimal number of lags: %i \n', rModelOpt.iNumLags);

% Restore path
path(sOldPath);
