function [rStatsOOS] = fEvaluatePerformanceOOS(mY,mYhatB,mYhatM)
% Function for calculating the OOS R2
%
% This function allows to analyze:
%   A: one true time-series and multiple predictions (mY is a vector and
%      mYhatB and mYhatM are matrices)
%   B: multiple time-series and one prediction for each (all mY, mYhatB, and
%      mYhatM are matrices)
%
% Output may differ
%   A: Statistics are a 1 x P vector, where P denotes the number of
%      predictions
%   B: Statistics are a scalar, i.e., all forecasts are pooled
%
% Input:
%   mY:         T x N matrix of observed values
%               T: number of time-series observations
%               N: number of objects
%   mYhatB:     T x N matrix of predicted values from the benchmark model
%               T: number of time-series observations
%               N: number of objects
%   mYhatM:     T x N matrix of predicted values from the model
%               T: number of time-series observations
%               N: number of objects
%
% Output:
%   rStatsOOS:  Struct, containing the results
%       .vR2OOS:        OOS R2
%       .vR2OOSCT:      OOS R2 wit Campbell/Thompson adjustment

% Determine dimensions
[iNumObs, iNumDepVars] = size(mY);
[iNumObsB, iNumPredB]  = size(mYhatB);
[iNumObsM, iNumPredM]  = size(mYhatM);

% Check dimensions
if iNumDepVars > 1
    % If multiple dependend variables are passed, the dimension must match
    % the number of predictions
    assert(iNumDepVars == iNumPredB, 'Number of time-series must agree');
    assert(iNumDepVars == iNumPredM, 'Number of time-series must agree');
else
    % Increase size of predictions
    mY = repmat(mY, 1, iNumPredM);
end
assert(iNumPredM == iNumPredB, 'Number of predictions must agree');
assert(iNumObs == iNumObsB, 'Number of observations must agree');
assert(iNumObs == iNumObsM, 'Number of observations must agree');

% Match missing values
lIsNaN = isnan(mY) | isnan(mYhatB) | isnan(mYhatM);
mY(lIsNaN)      = NaN;
mYhatB(lIsNaN)  = NaN;
mYhatM(lIsNaN)  = NaN;

% Get adjusted predictions
mYhatAdjB = max(mYhatB,0);
mYhatAdjM = max(mYhatM,0);

% Calculate errors
mErrorB     = (mY - mYhatB).^2;
mErrorM     = (mY - mYhatM).^2;
mErrorAdjB  = (mY - mYhatAdjB).^2;
mErrorAdjM  = (mY - mYhatAdjM).^2;

% Calculate R2
if iNumDepVars == 1
    % Only one dependent variable but multiple predictions
    rStatsOOS.vR2OOS    = 1 - (sum(mErrorM,1,'omitmissing')./sum(mErrorB,1,'omitmissing'));
    rStatsOOS.vR2OOSCT  = 1 - (sum(mErrorAdjM,1,'omitmissing')./sum(mErrorAdjB,1,'omitmissing'));
else
    % Only one dependent variable but multiple predictions
    rStatsOOS.vR2OOS    = 1 - (sum(mErrorM,'all','omitmissing')./sum(mErrorB,'all','omitmissing'));
    rStatsOOS.vR2OOSCT  = 1 - (sum(mErrorAdjM,'all','omitmissing')./sum(mErrorAdjB,'all','omitmissing'));
end

% Diebold-Mariano test
rStatsOOS.vDM       = NaN(1, iNumPredM);
rStatsOOS.vDMp      = NaN(1, iNumPredM);
rStatsOOS.vDM_CT    = NaN(1, iNumPredM);
rStatsOOS.vDMp_CT   = NaN(1, iNumPredM);
rStatsOOS.vCW       = NaN(1, iNumPredM);
rStatsOOS.vCWp      = NaN(1, iNumPredM);
rStatsOOS.vCW_CT    = NaN(1, iNumPredM);
rStatsOOS.vCWp_CT   = NaN(1, iNumPredM);
for iIdxP = 1:iNumPredM
    try
    % Diebold-Mariano test
    [rStatsOOS.vDM(iIdxP), rStatsOOS.vDMp(iIdxP)]       = fDieboldMariano(mY, mYhatB(:,iIdxP),mYhatM(:,iIdxP));
    [rStatsOOS.vDM_CT(iIdxP), rStatsOOS.vDMp_CT(iIdxP)] = fDieboldMariano(mY, mYhatAdjB(:,iIdxP),mYhatAdjM(:,iIdxP));

    % Clark-West test (perform only if R2 is positive)
    if rStatsOOS.vR2OOS(iIdxP) > 0
        [rStatsOOS.vCW(iIdxP), rStatsOOS.vCWp(iIdxP)]       = fClarkWest(mY, mYhatB(:,iIdxP),mYhatM(:,iIdxP));
    end
    if rStatsOOS.vR2OOSCT(iIdxP) > 0 
        [rStatsOOS.vCW_CT(iIdxP), rStatsOOS.vCWp_CT(iIdxP)] = fClarkWest(mY, mYhatAdjB(:,iIdxP),mYhatAdjM(:,iIdxP));
    end
    end
end
end