function rModel = fEstRNN(rModel, mY, mX, mWts)
% Function for estimating a recurrent neural network
%
% Input:
%   rCondNN:        Struct, containing the model settings and parameters
%   mY:             N x T matrix of the response variable
%                   N: number of responses
%                   T: number of time-series observations
%   mX:             L x T matrix of time-series variables
%                   L: number of time-series variables
%                   T: number of time-series observations
%   mWts:           N x T matrix of observation weights
%                   N: number of responses
%                   T: number of time-series observations

% Determine dimensions
[iNumResp, iNumObs] = size(mY);
[iNumVars, iNumObsX] = size(mX);

% Check dimensions
assert(iNumObs == iNumObsX,sprintf('Number of time-series observations (Y: %i, X: %i) does not agree',...
    iNumObs, iNumObsX));

% Optional arguments
if nargin == 4 && ~isempty(mWts)
    assert(iNumResp == size(mWts,1),...
        sprintf('Number of response variables (Y: %i, W: %i) does not agree',...
        iNumResp, size(mWts,1)));
    assert(iNumObs == size(mWts,2),...
        sprintf('Number of time-series observations (Y: %i, W: %i) does not agree',...
        iNumObs, size(mWts,2)));
else
    mWts = [];
end

% Control for missing values in mX
assert(any(~isnan(mX) | ~isinf(mX),"all"), 'mX must not contain missing values or inf');

% Save info
rModel.iNumResp = iNumResp;
rModel.iNumVars = iNumVars;
rModel.iNumObs = iNumObs;

% Set default parameters
rModel = fSetupRNN(rModel);

% Initialize parameters
rModel = fInitPrms(rModel);

% Optimize parameters
rModel = fOptRNN(rModel, mY, mX, mWts);
end

