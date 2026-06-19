 % Script for demonstrating the use of the RNN library and checking of the
 % gradient calculation
 
% Clear console
clear; clc; close all;

% Set paths
sOldPath = path;
addpath(genpath('./RNN'));

% Define data
iNumVars = 3;       % Exogeneous time-series variables
iNumResp = 10;       % Number of response variables
iNumObs = 50;       % Number of time-series observations
iNumUnits = 2;      % Number of units

% Draw random data
mX = randn(iNumVars, iNumObs);
mY = randn(iNumResp, iNumObs);

%% Settings for network
% === General architecture
rModel.iNumUnits = iNumUnits;           % Number of units

% === Hidden layer
rModel.rHidLayer.lEstBias = true;       % Estimate a bias
rModel.rHidLayer.iActivFun = 3;         % 3: tanh, 4: relu

% === Output layer
rModel.rOutLayer.lEstAlpha = false;      % Estimate intercept (bias)
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

% Other options
rModel = fSetupRNN(rModel);

%% Estimate model
% Estimate parameters
% rModel = fEstRNN(rModel, mY, mX);

%% Gradient checking
% Initialize network
rModel.iNumResp = iNumResp;
rModel.iNumVars = iNumVars;
rModel = fInitPrms(rModel);

% Parameters to vector
vPara0 = fParaMatToVec(rModel);

% Calculate gradient
[dOf, vGrad] = fObjFunRNN(rModel, vPara0, mY, mX, []);

% Numerical approximation
hFun = @(vPara)fObjFunRNN(rModel, vPara, mY, mX, []);
vGradApprox = fGradApprox(hFun, vPara0);

rGradApprox = fParaVecToMat(rModel, vGradApprox);
vGradDiff = vGrad - vGradApprox;

% Restore path
path(sOldPath);