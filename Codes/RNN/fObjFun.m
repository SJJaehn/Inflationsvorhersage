function dOf = fObjFun(mError, mWts, iObjFunType)
% Function for calculating the objective function value
%
% Input:
%   mError:         N x T matrix of errors
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
%   dOf:            Scalar, double, objective function value

% Objective function value
if (iObjFunType == 1)
    % Sum of squared residuals
    if isempty(mWts)
        vResid = mError(:);
        vResid(isnan(vResid)) = [];    
        dOf = vResid' * vResid;
    else
        dOf = sum( mWts .* (mError.^2) ,'all','omitnan');
    end
elseif (iObjFunType == 2)
    % Sum of absolute residuals
    if isempty(mWts)
        dOf = sum(abs(mError),'all','omitnan');
    else
        dOf = sum( mWts .* abs(mError) ,'all','omitnan');
    end
elseif (iObjFunType == 3)
    % Binary cross-entropy
    if isempty(mWts)
        dOf = sum(mError,'all','omitnan');
    else
        dOf = sum(mWts .* mError,'all','omitnan');
    end
else
    error('Unknown loss function');
end
end