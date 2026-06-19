function rModel = fEstOptVAR(mY, mX, options)
% Function for estimating an optimized vector autoregressive model using OLS.
%
% Input:
%   mY:         T x N matrix of dependent variables
%   mX:         T x K matrix of additional independent variables
%               (can be empty)
%   options:    Name-Value pair arguments
%       'vNumLags':     Vector of integers, number of lags of the dependent
%                       variable to be tested (default = 1:3)
%       'lEstAlpha':    Logical, specifies whether to estimate an intercept
%                       (default = true)
%
% Output:
%

% Check input arguments
arguments
    mY (:,:) {mustBeNumeric}
    mX (:,:) {mustBeNumeric} = []
    options.vNumLags (:,1) {mustBeNumeric, mustBeNonnegative} = 1:3
    options.iReportLag (1,1) {mustBeNumeric, mustBeNonnegative} = 0
    options.lEstAlpha (1,1) {mustBeNumericOrLogical} = true
end

% Determine dimensions
[iNumObs, iNumDepVars]  = size(mY);
iNumLagTest             = length(options.vNumLags);

% Check dimensions
if ~isempty(mX)
    [iNumObsX, iNumIndepVars] = size(mX);
    assert(iNumObs == iNumObsX,'Number of time-series dimensions must agree');
else
    % No additional predictors
    iNumIndepVars = 0;
end

% Initialize memory for AIC and estimated coefficients
vAIC    = NaN(iNumLagTest,1);
cModels = cell(iNumLagTest,1); 

% Loop over lags
for iIdxL = 1:iNumLagTest
    % Estimate model
    rModel = fEstVAR(mY, mX, 'iNumLags', options.vNumLags(iIdxL),...
        'iReportLag', options.iReportLag, 'lEstAlpha', options.lEstAlpha);

    % Get AIC
    vAIC(iIdxL)     = rModel.dAIC;
    cModels{iIdxL}  = rModel;
end

% Find best model
[~,idxBest] = min(vAIC);

% Get best model
rModel = cModels{idxBest};
end
