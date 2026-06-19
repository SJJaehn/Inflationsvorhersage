function [mYhat,mUnits,rModel] = fPredictRNN(rModel, mY, mX)
% Function for predicting using a trained RNN

%% Check inputs
% Determine dimension
[iNumResp, iNumTestObsY] = size(mY);
[iNumVars, iNumTestObs] = size(mX);

% Check dimensions
assert(iNumTestObs == iNumTestObsY, 'Number of observations must agree');

%% Predictions
% Preallocate memory
mYhat = NaN(iNumResp, iNumTestObs);
mUnits = NaN(rModel.iNumUnits, iNumTestObs);

% Iterate through time
for iIdxT = 1:iNumTestObs
    % Get previous hidden units
    if iIdxT == 1
        % Get last in-sample hidden units
        vUnitInit = rModel.mUnits(:,end);
        vErrInit = rModel.mInputEC(:,end);
    end
    
    % Forward pass
    [mYhat(:,iIdxT), vUnitInit, ~, vErrInit] = ...
        fForward(rModel, mY(:,iIdxT), mX(:,iIdxT), vUnitInit, vErrInit);
end
end