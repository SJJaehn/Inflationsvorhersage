% Script for demonstraing the usage of PCA. Now the PCs drive the variation
% of one response variable. Everything is estimated in-sample

% Clear console
clear; clc; close all;

% Set path
sOldPath = path;
addpath('./Utils');

% Random number generator
rng(42);

% Settings
iNumObs         = 1000;     % Number of observations
iNumVarsTrue    = 4;        % Number of true variables
iNumVarsNoise   = 200;      % Number of noisy variables
iNumDepVars     = 1;        % Number of dependent variables
dNoise          = 0.2;      % Noise factor

% Generate true components (with decreasing variance)
mPCs = randn(iNumObs, iNumVarsTrue) .* linspace(1,0.5,iNumVarsTrue);

% Generate noisy variables
mLoadings = randn(iNumVarsNoise, iNumVarsTrue);
mX        = mPCs * mLoadings' + dNoise * randn(iNumObs, iNumVarsNoise);

% Generate response variable
mBeta     = randn(iNumDepVars, iNumVarsTrue); % True betas on PCs
mY        = mPCs * mBeta' + dNoise * randn(iNumObs, iNumDepVars);

% Linear regression with noisy predictors (with constant)
mBetaOLS  = [ones(iNumObs,1), mX]\mY;
mYhatOLS  = [ones(iNumObs,1), mX] * mBetaOLS;
dMSE_OLS  = mean( (mY - mYhatOLS).^2,'all');
fprintf('RMSE OLS %.4f \n',sqrt(dMSE_OLS));

% Estimate full PCA with 90% explained variance
rModel  = fEstPCA(mX, "dFracVar", 0.9, 'iTransformX',1);
mScores = rModel.mScores;

% Linear regression with PCs (with constant)
mBetaPCR  = [ones(iNumObs,1), mScores]\mY;
mYhatPCR  = [ones(iNumObs,1), mScores] * mBetaPCR;
dMSE_PCR  = mean( (mY - mYhatPCR).^2,'all');
fprintf('RMSE PCR %.4f \n',sqrt(dMSE_PCR));

% Restore path
path(sOldPath);