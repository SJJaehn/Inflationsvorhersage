% Script for demonstrating the implementation of a VAR model with exogenous
% predictors

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
iNumIndepVars   = 2;        % Number of independent variables

% Simulate Phi matrix
mPhi = cat(3, [0.2, 0.0; -0.3, 0.4],... % First lag
    [-0.1, 0.1; 0.2, -0.3]);

% Determine dimensions
[~,iNumSeries,iNumLags] = size(mPhi);

% Initialize exogenous predictors
mX      = randn(iNumObs + iNumInit, iNumIndepVars);
mTheta  = [0.5, 0;0, -0.5];

% Initialize random shocks
mY                  = randn(iNumObs+iNumInit,iNumSeries) * 0.5;

% Generate data
for iIdxT = (iNumLags+1):(iNumObs+iNumInit)
    % Get the component of y due to random noise
    vYnoise = mY(iIdxT,:)';

    % Get the component of y due to lags
    vYlag   = zeros(iNumSeries,1);
    for iIdxP = 1:iNumLags
        vYlag = vYlag + mPhi(:,:,iIdxP) * mY(iIdxT-iIdxP,:)';
    end

    % Get the component of y due to X
    vYx = mTheta * mX(iIdxT,:)';

    % Add together
    mY(iIdxT,:) = vYnoise + vYlag + vYx;
end

% Remove initialization period
mY(1:iNumInit,:) = [];
mX(1:iNumInit,:) = [];

%% Estimation
% Estimate VAR(1) model
rModelVAR1 = fEstVAR(mY, mX, 'iNumLags', 1, 'lEstAlpha', true);

% Estimate VAR(2) model
rModelVAR2 = fEstVAR(mY, mX, 'iNumLags', 2, 'lEstAlpha', true);

% Get difference in estimated coefficients
mDiff       = rModelVAR2.mPhi - mPhi;
mDiffTheta  = rModelVAR2.mTheta - mTheta;
fprintf('Mean absolute difference in phi coefficients: %.4f \n', mean(abs(mDiff),'all'))
fprintf('Mean absolute difference in theta coefficients: %.4f \n', mean(abs(mDiffTheta),'all'))

% Select best lag using AIC
rModelOpt = fEstOptVAR(mY, mX);
fprintf('Optimal number of lags: %i \n', rModelOpt.iNumLags);

% Restore path
path(sOldPath);