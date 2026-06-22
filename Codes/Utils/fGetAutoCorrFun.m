function [mCorrCoef, mCorrCoefP] = fGetAutoCorrFun(vY, mX, iMaxLag)
% Function for obtaining the ACF
%
% Input:
%   vY:         T x 1 vector of the dependent variable
%               T: number of time-series observations
%   mX:         T x K matrix of independent variables. 
%               If empty, the ACF of vY with itself is calculated (K = 1)
%               T: number of time-series observations
%               K: number of independent variables
%   iMaxLag:    Scalar, integer, maximum number of lags
%               (default = 20)
%
% Output:
%   mCorrCoef:  K x iMaxLag matrix of (auto)correlation coefficients is
%               returned
%   mCorrCoefP: K x iMaxLag matrix of p-values (auto)correlation coefficients 
%               is returned

% Check input arguments
arguments
    vY (:,1)
    mX (:,:) = []
    iMaxLag (1,1) {mustBeNumeric, mustBeNonnegative} = 20
end

% Set dependent variable to independent variable if no mX passed
if isempty(mX)
    mX = vY;
end

% Determine dimensions
[iNumObs, iNumIndepVars] = size(mX);

% Initialize memory
mCorrCoef  = NaN(iNumIndepVars, iMaxLag);
mCorrCoefP = NaN(iNumIndepVars, iMaxLag);

% Loop over lags
for iIdxL = 1:iMaxLag
    % Lag data
    mXlag = [NaN(iIdxL, iNumIndepVars); mX(1:end-iIdxL,:)];

    % Loop over variables
    for iIdxJ = 1:iNumIndepVars
        % Copy data
        vYtemp = vY;
        vXtemp = mXlag(:,iIdxJ);

        % Remove missing values
        lIsNaN          = isnan(vYtemp) | isnan(vXtemp);
        vYtemp(lIsNaN)  = [];
        vXtemp(lIsNaN)  = [];

        % Calculate correlation
        [mCorrCoef(iIdxJ, iIdxL), mCorrCoefP(iIdxJ, iIdxL)] = ...
            corr(vYtemp, vXtemp);
    end   
end
end

