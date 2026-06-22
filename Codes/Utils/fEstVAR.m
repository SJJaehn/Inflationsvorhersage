function rModel = fEstVAR(mY, mX, options)
% Function for estimating a vector autoregressive model using OLS.
%
% Input:
%   mY:         T x N matrix of dependent variables
%   mX:         T x K matrix of additional independent variables
%               (can be empty)
%   options:    Name-Value pair arguments
%       'iNumLags':     Scalar, integer, number of lags of the dependent
%                       variable
%       'lEstAlpha':    Logical, specifies whether to estimate an intercept
%                       (default = true)
%
% Output:
%   rModel:     Struct, containing model settings and parameters
%       .iNumLags:      Scalar, integer, number of lags
%       .lEstAlpha:     Logical, specifies whether to estimate an intercept
%       .iNumDepVars:   Scalar, integer, number of dependent variables
%       .iNumindepVars: Scalar, integer, number of independent variables
%       .iNumPrms:      Scalar, integer, number of estimated parameters
%       .dAIC:          Scalar, double, AIC
%       .mCoef:         (lEstAlpha + N * iNumLags + K) x N matrix of 
%                       regression coefficients
%       .vAlpha:        1 x N vector of the regression intercepts
%       .mSigma:        N x N x iNumLags matrix of VAR coefficients
%       .mTheta:        K x N matrix of regression coefficients of the
%                       exogenous regressors

% Check input arguments
arguments
    mY (:,:) {mustBeNumeric}
    mX (:,:) {mustBeNumeric} = []
    options.iNumLags (1,1) {mustBeNumeric, mustBeNonnegative} = 1
    options.lEstAlpha (1,1) {mustBeNumericOrLogical} = true
end

% Determine dimensions
[iNumObs, iNumDepVars] = size(mY);

% Check dimensions
if ~isempty(mX)
    [iNumObsX, iNumIndepVars] = size(mX);
    assert(iNumObs == iNumObsX,'Number of time-series dimensions must agree');
else
    % No additional predictors
    iNumIndepVars = 0;
end

% Save information regarding model
rModel.iNumLags     = options.iNumLags;
rModel.lEstAlpha    = options.lEstAlpha;
rModel.iNumDepVars  = iNumDepVars;
rModel.iNumIndepVars= iNumIndepVars;

% Lag time-series
mYlag = lagmatrix(mY,1:options.iNumLags);

% Find missing values
lIsNaN          = any(isnan(mYlag),2) | any(isnan(mY),2);
if ~isempty(mX)
    lIsNaN = lIsNaN | any(isnan(mX),2);
    mX(lIsNaN,:) = [];
end
% Remove missing values
mYlag(lIsNaN,:)  = [];
mY(lIsNaN,:)     = [];

% Number of remaining observations
iNumObs = size(mY,1);

% Get all regressors
mXreg = [ones(iNumObs, options.lEstAlpha), mYlag, mX];

% Regression. The first column contains all coefficients that belong to the
% first dependent variable, etc.
mBeta = mXreg\mY;

% Get fitted values
mYhat = mXreg * mBeta;

% Get residuals
mResid = mY - mYhat;

% Calculate AIC
    % Compute residual covariance matrix (Σ)
mSigma   = (mResid' * mResid) / (iNumObs - options.iNumLags - iNumDepVars);

    % Number of estimated parameters
iNumPrms = numel(mBeta);

    % Compute AIC
dAIC     = 2 * iNumPrms + (iNumObs - options.iNumLags) * log(det(mSigma));

% Save coefficients, residual covariance matrix, and AIC
rModel.iNumPrms = iNumPrms;
rModel.dAIC     = dAIC;
rModel.mCoef    = mBeta;
rModel.mSigma   = mSigma;

% Decompose regression coefficient matrix
if options.lEstAlpha
    rModel.vAlpha      = mBeta(1,:);
    mBeta(1,:)  = [];
else
    rModel.vAlpha      = zeros(1, iNumDepVars);
end

% Get coefficients of exogeneous regressors
if ~isempty(mX)
    rModel.mTheta = mBeta(end-iNumIndepVars+1:end,:)';
    mBeta(end-iNumIndepVars+1:end,:) = [];
else
    rModel.mTheta = [];
end

% The remaining coefficients are VAR coefficients. Reshape into N x N x L
% matrix, where L is the number of lags.
rModel.mPhi = reshape(permute(mBeta,[2,1,3]),iNumDepVars,iNumDepVars,options.iNumLags);

% The third dimension indicates the lag, i.e., the first slice is the first lag

% The rows indicate the coefficients for each variable, i.e., the
% coefficient in the first row and second column measures how the first
% variable is influenced by the lags of the second variable

% Special case: AR model (remove second dimension)
if iNumDepVars == 1
    rModel.mPhi = squeeze(rModel.mPhi);
end

% Save dependent variables
rModel.mY        = mY;
end