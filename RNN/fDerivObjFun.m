function mDeriv = fDerivObjFun(mY, mYhat, mWts, iObjFunType)
% Function for calculating the derivative of the objective function value
%
% Input:
%   mY:             N x T matrix of observations for the response variables
%                   N: number of responses
%                   T: number of time-series observations
%   mYhat:          N x T matrix of estimates for the response variables
%                   N: number of responses
%                   T: number of time-series observations
%   mWts:           N x T matrix of observation weights
%                   N: number of responses
%                   T: number of time-series observations
%   iObjFunType:    Scalar, integer, objective function type
%                   1: sum of squared residuals
%                   2: sum of absolute residuals
%                   3: binary cross-entropy
%
% Output:
%   mDeriv:         N x T matrix of derivatives
%                   N: number of responses
%                   T: number of time-series observations

% Objective function value
if (iObjFunType == 1)
    % Sum of squared residuals
    if isempty(mWts)
        mDeriv = - (2 * (mY - mYhat));
    else
        mDeriv = - (2 * mWts .* (mY - mYhat));
    end
elseif (iObjFunType == 2)
    % Sum of absolute residuals
    if isempty(mWts)
        mDeriv = sign( mYhat - mY );
    else
        mDeriv = sign( mWts .* (mYhat - mY ));
%         mDeriv = sign( mYhat - mY );
    end
elseif (iObjFunType == 3)
    % Binary cross-entropy
    dEps = 1e-10;
    if isempty(mWts)
        mDeriv = ((- mY ./(mYhat + dEps)) + ((1 - mY)./(1 - mYhat + dEps)));
    else
        error('Needs to be checked');
    end
else
    error('Unknown loss function');
end
end