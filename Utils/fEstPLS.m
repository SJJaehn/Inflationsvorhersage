function rModel = fEstPLS(mY, mX, options)
% Function for estimating a partial least squares regression
%
% Input:
%   mY:         N x M matrix of dependent variables
%               N: number of observations
%               M: number of dependent variables
%   mX:         N x P matrix of independent variables
%               N: number of observations
%               P: number of independent variables
%   options:    Name-Value pair arguments
%       'iNumComp':     Scalar, integer, number of components (default = 1).
%                       Note that the input is ignored if 'dFracVar' is
%                       nonmissing
%       'dFracVar':     Scalar, double, specifies the fraction of explained
%                       variance (default = NaN)
%       'iMaxComp':     Scalar, integer, maximum number of PLS components
%                       (default = max(M, P)-1)
%       'iTransformX':  Scalar, integer, specifies whether to transform the
%                       data
%                       0: no transformation
%                       1: centering (default)
%                       2: z-transformation
%       'lPositiveMean':Logical, specifies whether to ensure that scores
%                       have positive mean. May be useful in some empirical 
%                       applications
%
% Output:
%   rModel:     Struct, containing the following fields
%       .iNumComp:      Scalar, integer, number of estimated principal
%                       components
%       .vMeanX:        1 x P vector of column means
%                       P: number of variables
%       .vStdX:         1 x P vector of column standard deviations
%                       P: number of variables
%       .mEigVec:       P x iNumComp matrix of eigenvectors
%                       P: number of variables
%                       iNumComp: number of PCs
%       .vEigVal:       iNumComp x 1 vector of eigenvalues
%                       iNumComp: number of PCs
%       .mScores:       N x iNumComp matrix of scores
%                       N: number of observations
%                       iNumComp: number of PCs
%       .vExplVar:      iNumComp x 1 vector of cumulative variance
%                       explained by the PCs

% Check input arguments
arguments 
    mY (:,:) {mustBeNumeric}
    mX (:,:) {mustBeNumeric}
    options.iNumComp (1,1) {mustBeNumeric, mustBeNonnegative} = 1
    options.dFracVar (1,1) {mustBeNumeric} = NaN
    options.iMaxComp (1,1) {mustBeNumeric} = NaN
    options.iTransformX (1,1) {mustBeNumeric, mustBeNonnegative} = 1
    options.iTransformY (1,1) {mustBeNumeric, mustBeNonnegative} = 1
    options.lPositiveMean (1,1) {mustBeNumericOrLogical} = false
end

% Check data
assert(~any(isnan(mX),'all'),'No missing values allowed');

% Determine dimensions
[iNumObs, iNumDepVars]    = size(mY);
[iNumObsX, iNumIndepVars] = size(mX);

% Check dimensions
assert(iNumObs == iNumObsX, 'Number of time-series observations must agree');

% Maximum number of components to extract
if isnan(options.iMaxComp)
    options.iMaxComp = min(max(iNumIndepVars, iNumDepVars), iNumObs)-1;
end

% Transform data
if options.iTransformX == 0
    % No transformation
    rModel.vMeanX = zeros(1, iNumIndepVars);
    rModel.vStdX  = ones(1, iNumIndepVars);
elseif options.iTransformX == 1
    % Center data
    rModel.vMeanX = mean(mX,1);
    rModel.vStdX  = ones(1, iNumIndepVars);
elseif options.iTransformX == 2
    % z-transformation
    rModel.vMeanX = mean(mX,1);
    rModel.vStdX  = std(mX,[],1);
else
    error('Unknown transformation method');
end

% Transform dependent data
if options.iTransformY == 0
    % No transformation
    rModel.vMeanY = zeros(1, iNumDepVars);
    rModel.vStdY  = ones(1, iNumDepVars);
elseif options.iTransformY == 1
    % Center data
    rModel.vMeanY = mean(mY,1);
    rModel.vStdY  = ones(1, iNumDepVars);
elseif options.iTransformY == 2
    % z-transformation
    rModel.vMeanY = mean(mY,1);
    rModel.vStdY  = std(mY,[],1);
else
    error('Unknown transformation method');
end

% Standardize data if requested
mX = (mX - rModel.vMeanX)./rModel.vStdX;
mY = (mY - rModel.vMeanY)./rModel.vStdY;

% Estimate PLS
[~,~,~,~,~,pctVar,~,stats] = plsregress(mX, mY, options.iMaxComp);

% The weights that are required for constructing the scores from X are in
% stats.W
mWeightsX = stats.W;

% Get the scores
mScores = fProjectData(mX, mWeightsX);

% In some empirical application, we want to require the scores to have a 
% positive mean
if options.lPositiveMean
    vSign               = sign(mean(mScores,1));
    vSign(vSign == 0)   = 1;
    mScores             = mScores .* vSign;
    mWeightsX           = mWeightsX .* vSign;
end

% Get fraction of explained variance (1. ros: X, 2. row: Y)
mCumVarExpl = cumsum(pctVar,2);

% Get number of PCs to keep
if isnan(options.dFracVar)
    % Keep the prespecified number of PCs
    iNumComp = options.iNumComp;
else
    % Get number of PCs that explains sufficient variance of both X and Y
    iNumComp = find(all(mCumVarExpl >= options.dFracVar,1), 1, 'first');
end

% Dimensionality reduction
rModel.iNumComp  = iNumComp;
rModel.mWeightsX = mWeightsX(:,1:iNumComp);
rModel.mScores   = mScores(:,1:iNumComp);
rModel.mExplVar  = mCumVarExpl(:,1:iNumComp);
end