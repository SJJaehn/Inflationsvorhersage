% Script for testing the RNN code - out-of-sample

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
mAR    = 0.1 * randn(iNumUnits, iNumUnits);   % (hidden -> hidden)
mOmega = 0.1 * randn(iNumVars, iNumUnits);    % (input -> hidden)
mBeta  = 0.1 * randn(iNumResp, iNumUnits);    % (hidden -> output)

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
mY = mBeta' * mUnits;

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
rModel.rOptimOpt.sDisplay = 'off';     % 'iter' or 'off'
rModel.rOptimOpt.iMaxTime = Inf;        % Time limit
rModel.rOptimOpt.vLB = -100;            % Lower bounds for parameters
rModel.rOptimOpt.vUB = 100;             % Upper bounds for parameters

%% Out-of-sample testing
iNumIn = 80;            % Number of in-sample periods
iNumOut = 1;            % Number of out-of-sample periods
lRoll = false;          % Rolling (true) or expanding (false) time window

% Initialize memory
mYhatMean = NaN(iNumResp, iNumObs);
mYhatLinReg = NaN(iNumResp, iNumObs);
mYhatRNN = NaN(iNumResp, iNumObs);

% Iterate through time
for iIdxT = iNumIn:iNumOut:iNumObs-1
    % Get in-sample and out-of-sample indices
    if lRoll 
        vIdxInSample = (iIdxT-iNumIn+1):iIdxT;
    else
        vIdxInSample = 1:iIdxT;
    end
    vIdxOutOfSample = (iIdxT+1):(iIdxT+iNumOut);
    vIdxOutOfSample(vIdxOutOfSample > iNumObs) = [];
    
    % Get training data
    mYin = mY(:,vIdxInSample);
    mXin = mX(:,vIdxInSample);
    
    % Get out-of-sample data
    mYout = mY(:,vIdxOutOfSample);
    mXout = mX(:,vIdxOutOfSample);
    
    % === Mean model
    mYhatMean(:,vIdxOutOfSample) = repmat(mean(mYin,2),1,length(vIdxOutOfSample));
    
    % === Linear regression
    vBetaHat = regress(mYin', mXin');
    mYhatLinReg(:,vIdxOutOfSample) = vBetaHat' * mXout;
    
    % === RNN
    % Estimate model
    rModelTrained = fEstRNN(rModel, mYin, mXin, []);
    
    % Prediction
    mYhatRNN(:,vIdxOutOfSample) = fPredictRNN(rModelTrained, mYout, mXout);
end  
    
% Evaluate the quality
dMSE_Mean = sum( (mY - mYhatMean).^2, 'all', 'omitnan');
dMSE_LinReg = sum( (mY - mYhatLinReg).^2, 'all','omitnan');
dR2_LinReg = 1 - (dMSE_LinReg/dMSE_Mean);
dMSE_RNN = sum( (mY - mYhatRNN).^2, 'all','omitnan');
dR2_RNN = 1 - (dMSE_RNN/dMSE_Mean);
