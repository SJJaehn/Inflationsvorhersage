function rModel = fEstPCA(mX, options)
% Function for estimating a principal component analysis 
%
% Input:
%   mX:         N x P data matrix
%               N: number of observations
%               P: number of variables
%   options:    Name-Value pair arguments
%       'iNumComp':     Scalar, integer, number of components (default = 1).
%                       Note that the input is ignored if 'dFracVar' is
%                       nonmissing
%       'dFracVar':     Scalar, double, specifies the fraction of explained
%                       variance (default = NaN)
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
    mX (:,:) {mustBeNumeric}
    options.iNumComp (1,1) {mustBeNumeric, mustBeNonnegative} = 1
    options.dFracVar (1,1) {mustBeNumeric} = NaN
    options.iTransformX (1,1) {mustBeNumeric, mustBeNonnegative} = 1
    options.lPositiveMean (1,1) {mustBeNumericOrLogical} = false
end

% Check data
assert(~any(isnan(mX),'all'),'No missing values allowed');

% Determine dimensions
[iNumObs, iNumVariables] = size(mX);

% Transform data
if options.iTransformX == 0
    % No transformation
    rModel.vMeanX = zeros(1, iNumVariables);
    rModel.vStdX  = ones(1, iNumVariables);
elseif options.iTransformX == 1
    % Center data
    rModel.vMeanX = mean(mX,1);
    rModel.vStdX  = ones(1, iNumVariables);
elseif options.iTransformX == 2
    % z-transformation
    rModel.vMeanX = mean(mX,1);
    rModel.vStdX  = std(mX,[],1);
else
    error('Unknown transformation method');
end

% Standardize data if requested
mX = (mX - rModel.vMeanX)./rModel.vStdX;
    
% Calculate covariance matrix
mSigma = cov(mX);

% Eigenvalue decomposition
[mEigVec, vEigVal] = eig(mSigma,'vector');  % Vector ensures that vEigVal is a vector, not a diagonal matrix

% Sort by eigenvalues
[vEigVal, vSortIdx] = sort(vEigVal,'descend');
mEigVec = mEigVec(:,vSortIdx);

% Get scores
mScores = fProjectData(mX, mEigVec); % Get scores

% In some empirical application, we want to require the scores to have a 
% positive mean
if options.lPositiveMean
    vSign               = sign(mean(mScores,1));
    vSign(vSign == 0)   = 1;
    mScores             = mScores .* vSign;
    mEigVec             = mEigVec .* vSign;
end

% Get fraction of explained variance
vExplVar = cumsum(vEigVal)./sum(vEigVal);

% Get number of PCs to keep
if isnan(options.dFracVar)
    % Keep the prespecified number of PCs
    iNumComp = options.iNumComp;
else
    % Get number of PCs that explains sufficient variance
    iNumComp = find(vExplVar >= options.dFracVar, 1, 'first');
end

% Dimensionality reduction
rModel.iNumComp = iNumComp;
rModel.mEigVec  = mEigVec(:,1:iNumComp);
rModel.vEigVal  = vEigVal(1:iNumComp);
rModel.mScores  = mScores(:,1:iNumComp);
rModel.vExplVar = vExplVar(1:iNumComp);
end