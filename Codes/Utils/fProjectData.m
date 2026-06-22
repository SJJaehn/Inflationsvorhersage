function mScores = fProjectData(mX, mEigVec, vMeanX, vStdX)
% Function for projecting data onto new axes
%
% Input:
%   mX:             N x P data matrix
%                   N: number of observations
%                   P: number of variables
%   mEigVec:        P x K matrix of eigenvectors
%                   P: number of variables
%                   K: number of PCs
%   vMeanX:         1 x P vector of column means
%                   P: number of variables
%   vStdX:          1 x P vector of column standard deviations
%                   P: number of variables
% Output:
%   mScores:        N x K matrix of scores
%                   N: number of observations
%                   K: number of PCs

% Determine dimensions
[iNumObs, iNumVariables] = size(mX);

% Check inputs
if nargin < 4 || isempty(vStdX)
    vStdX = ones(1, iNumVariables);
end
if nargin < 3 || isempty(vMeanX)
    vMeanX = zeros(1, iNumVariables);
end

% Standardize data if requested
mX = (mX - vMeanX)./vStdX;

% Get scores
mScores = mX * mEigVec;
end