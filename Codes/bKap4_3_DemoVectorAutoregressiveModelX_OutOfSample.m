% Script for demonstrating the implementation of a VAR model - out-of-sample

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

    % Add together
    mY(iIdxT,:) = vYnoise + vYlag;
end

% Remove initialization period
mY(1:iNumInit,:) = [];

%% Out-of-sample analysis
% Settings
iNumIn = 5000;
lRoll  = false;

% Initialize memory
mYhatVAR1       = NaN(iNumObs, iNumSeries);
mYhatVAR_Opt    = NaN(iNumObs, iNumSeries);

% Loop over time
for iIdxT = iNumIn:iNumObs-1
    % Get index
    if lRoll
        vIdxInSample = (iIdxT-iNumIn+1):iIdxT;
    else
        vIdxInSample = 1:iIdxT;
    end
    vIdxOutOfSample = iIdxT + 1;

    % Get data
    mYin = mY(vIdxInSample,:);

    % Estimate VAR(1) model
    rModelVAR1 = fEstVAR(mYin, [], 'iNumLags', 1, 'lEstAlpha', true);

    % Prediction
    mYhatVAR1(vIdxOutOfSample,:) = fPredictVAR(rModelVAR1, 1, []);

    % Estimate optimal VAR
    rModelOpt = fEstOptVAR(mYin, []);
    
    % Prediction
    mYhatVAR_Opt(vIdxOutOfSample,:) = fPredictVAR(rModelOpt, 1, []);
end

% Performance evaluation
dMSE_VAR1       = mean( (mY - mYhatVAR1).^2,'all','omitmissing');
dMSE_VAR_Opt    = mean( (mY - mYhatVAR_Opt).^2,'all','omitmissing');

% Print progress
fprintf('RMSE VAR(1) %.4f \n',sqrt(dMSE_VAR1));
fprintf('RMSE VAR-Opt %.4f \n',sqrt(dMSE_VAR_Opt));

% Restore path
path(sOldPath);