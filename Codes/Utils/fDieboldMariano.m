function [dTestStat, dPval] = fDieboldMariano(mY, mYhat1, mYhat2)
% Function for performing the modified Diebold-Mariano (1995) test
% Allows to analyze large cross-sections following Gu, Kelly, Xiu (2020)
%
% Input:
%   mYtrue:         T x N matrix of the true observations
%                   T: number of time-series observations
%                   N: number of response variables
%   mYhat1:         T x N matrix of the predictions of model 1
%                   T: number of time-series observations
%                   N: number of response variables
%   mYhat2:         T x N matrix of the predictions of model 2
%                   T: number of time-series observations
%                   N: number of response variables
%
% Output:
%   dPval:          Scalar, double, p-value
%   dTestStat:      Scalar, double, test statistic

%% Check inputs
% Determine dimensions
[iNumObs, iNumVars] = size(mYhat1);
[iNumObs2, iNumVars2] = size(mYhat2);

% Check dimensions
assert(iNumObs == iNumObs2, 'Number of observations must agree');
assert(iNumVars == iNumVars2, 'Number of variables must agree');

%% Diebold-Mariano-Test
% Ensure that same predictions are compared
lIsNaN          = isnan(mYhat1) | isnan(mYhat2);
mYhat1(lIsNaN)  = NaN;
mYhat2(lIsNaN)  = NaN;

% Remove only-NaN time steps
lNaN            = all(isnan(mYhat1),2) | all(isnan(mYhat2),2);
mYhat1(lNaN,:)  = [];
mYhat2(lNaN,:)  = [];
mY(lNaN,:)      = [];

% Update number of time-series observations
iNumObs = size(mY,1);

% Calculate the residuals
mResid1 = mY - mYhat1;
mResid2 = mY - mYhat2;
    
% Calculate average difference of squared errors
vErrorDiff = mean( (mResid1.^2 - mResid2.^2) , 2, 'omitnan');

% Regression on constant to get the test statistic
mX = [ones(iNumObs,1)];
rRegResults = regstats2(vErrorDiff, mX, 'onlydata',{'tstat','hac'});

% Save results
dTestStat  = rRegResults.tstat.beta ./ rRegResults.tstat.se;

% Adjust test statistic (Harvey, Leybourne, and Newbold (1997))
iNumLags = 1; % 1-period ahead forecast
dK = sqrt((iNumObs+1-2*iNumLags+(((iNumLags)*(iNumLags-1))/iNumObs))/iNumObs); 
dTestStat = dTestStat * dK;

% Degrees of freedom
df = iNumObs - size(mX,2);

% Calculate p-value
dPval = 2 * tcdf(-abs(dTestStat),df);      % two-sided
% dPval = 1 - tcdf(rResults.vBetaT,rResults.df);            % one-sided
end