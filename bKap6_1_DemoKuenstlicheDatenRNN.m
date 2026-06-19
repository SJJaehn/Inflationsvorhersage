% Script for testing the RNN code using artificial data

% Clear console
clear; clc; close all;

% Set paths
sOldPath = path;
addpath(genpath('./RNN'));

% Define data
iNumVars = 5;           % Exogeneous time-series variables
iNumResp = 1;           % Number of response variables
iNumObs = 100;          % Number of time-series observations
iNumUnits = 1;          % Number of units

% Generate weight matrices
mAR = 0.1 * randn(iNumUnits, iNumUnits);   % (hidden -> hidden)
mOmega = 0.1 * randn(iNumVars, iNumUnits); % (input -> hidden)
mBeta = 0.1 * randn(iNumResp, iNumUnits);  % (hidden -> output)

% Generate data
mX = randn(iNumVars, iNumObs);      % Predictor variables
mUnits = NaN(iNumUnits, iNumObs);
for iIdxT = 1:iNumObs
    if iIdxT == 1
        mUnits(:,iIdxT) = tanh(mOmega' * mX(:,iIdxT));
    else
        mUnits(:,iIdxT) = tanh(mOmega' * mX(:,iIdxT) + ...
            mAR' * mUnits(:,iIdxT-1));
    end
end
mY = mBeta * mUnits;

%% Test benchmark model - historical mean
mYhatMean = repmat(mean(mY,2),1, iNumObs);
dMSE_Mean = sum( (mY - mYhatMean).^2, 'all');

%% Test linear regression (without intercept)
vBetaHat = regress(mY', mX');
mYhatLinReg = vBetaHat' * mX;
dMSE_LinReg = sum( (mY - mYhatLinReg).^2, 'all');
dR2_LinReg = 1 - (dMSE_LinReg/dMSE_Mean);

%% Settings for network
% === General architecture
rModel.iNumUnits = iNumUnits;           % Number of units

% === Hidden layer
rModel.rHidLayer.lEstBias = true;       % Estimate a bias
rModel.rHidLayer.iActivFun = 3;         % 3: tanh, 4: relu

% === Output layer
rModel.rOutLayer.lEstAlpha = false;     % Estimate intercept (bias)
rModel.rOutLayer.iActivFun = 1;         % 1: linear, 2: sigmoid, 3: tanh, 4: relu

% === Error correction layer
rModel.rErrCorrLayer.iMode = 1;         % 1: error is input to model, 2: last estimate is input to model, 3: last realization is input
rModel.rErrCorrLayer.iNumUnits = 0;     % Number of units in error correction network
rModel.rErrCorrLayer.iActivFun = 3;     % 1: linear, 2: sigmoid, 3: tanh, 4: relu

% === Optimization
rModel.rOptimOpt.iObjFun = 1;           % 1: SSR, 2: MSE, 3: Cross entropy (classification)
rModel.rOptimOpt.sSolver = 'rprob';     % Solver (adam, nadam, rprob)
rModel.rOptimOpt.iMaxIter = 500;        % Maximum number of iterations
rModel.rOptimOpt.sDisplay = 'iter';     % 'iter' or 'off'
rModel.rOptimOpt.iMaxTime = Inf;        % Time limit
rModel.rOptimOpt.vLB = -100;            % Lower bounds for parameters
rModel.rOptimOpt.vUB = 100;             % Upper bounds for parameters

% Train RNN
rModel = fEstRNN(rModel, mY, mX, []);

% In-sample test
mYhatRNN = rModel.mYhat;
dMSE_RNN = sum( (mY - mYhatRNN).^2, 'all');
dR2_RNN  = 1 - (dMSE_RNN/dMSE_Mean);

% Restore path
path(sOldPath);