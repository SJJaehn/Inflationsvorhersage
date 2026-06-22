% Script for demonstrating the usage of PLS.

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
iNumVarsNoise   = 250;      % Number of noisy variables
iNumDepVars     = 1;        % Number of dependent variables
dNoise          = 0.2;      % Noise factor

% Out-of-sample design
iNumIn          = fix(iNumObs/2);
lRoll           = true;

% Generate true components (with decreasing variance)
mPCs = randn(iNumObs, iNumVarsTrue) .* linspace(1,0.5,iNumVarsTrue);

% Generate noisy variables
mLoadings = randn(iNumVarsNoise, iNumVarsTrue);
mX        = mPCs * mLoadings' + dNoise * randn(iNumObs, iNumVarsNoise);

% Generate response variable
mBeta     = randn(iNumDepVars, iNumVarsTrue); % True betas on PCs
mY        = mPCs * mBeta' + dNoise * randn(iNumObs, iNumDepVars);

% Initialize memory
mYhatOLS  = NaN(iNumObs, iNumDepVars);
mYhatPCR  = NaN(iNumObs, iNumDepVars);
mYhatPLS  = NaN(iNumObs, iNumDepVars);

% Loop over time
for iIdxT = iNumIn:iNumObs-1
    % Get in-sample index
    if lRoll
        vIdxInSample = (iIdxT-iNumIn+1):iIdxT;
    else 
        vIdxInSample = 1:iIdxT;
    end
    
    % Get data
    mXin  = mX(vIdxInSample,:);
    mXout = mX(iIdxT+1,:);
    mYin  = mY(vIdxInSample,:);

    % Determine dimensions
    iNumObsIn  = size(mXin,1);
    iNumObsOut = size(mXout,1);

    % Linear regression with noisy predictors (with constant)
    mBetaOLS            = [ones(iNumObsIn, 1), mXin]\mYin;
    mYhatOLS(iIdxT+1,:) = [ones(iNumObsOut, 1),mXout] * mBetaOLS;    

    % Estimate PCA with 90% explained variance
    rModel     = fEstPCA(mXin, "dFracVar", 0.9);
    mScoresIn  = rModel.mScores;
    mScoresOut = fProjectData(mXout, rModel.mEigVec, rModel.vMeanX, rModel.vStdX);

    % Linear regression with PCs (with constant)
    mBetaPCR            = [ones(iNumObsIn,1),mScoresIn]\mYin;
    mYhatPCR(iIdxT+1,:) = [ones(iNumObsOut,1), mScoresOut] * mBetaPCR;

    % Estimate PLS
    rModel      = fEstPLS(mYin, mXin, "dFracVar", 0.9);
    mScoresIn   = rModel.mScores;
    mScoresOut  = fProjectData(mXout, rModel.mWeightsX, rModel.vMeanX, rModel.vStdX);
    
    % Linear regression with PLS-PCs (with constant)
    mBetaPLS            = [ones(iNumObsIn,1), mScoresIn]\mYin;
    mYhatPLS(iIdxT+1,:) = [ones(iNumObsOut,1),mScoresOut] * mBetaPLS;
end

% Evaluate performance
dMSE_OLS  = mean( (mY - mYhatOLS).^2,'all','omitmissing');
fprintf('RMSE OLS %.4f \n',sqrt(dMSE_OLS));
dMSE_PCR  = mean( (mY - mYhatPCR).^2,'all','omitmissing');
fprintf('RMSE PCR %.4f \n',sqrt(dMSE_PCR));
dMSE_PLS  = mean( (mY - mYhatPLS).^2,'all','omitmissing');
fprintf('RMSE PLS %.4f \n',sqrt(dMSE_PLS));

% Restore path
path(sOldPath);