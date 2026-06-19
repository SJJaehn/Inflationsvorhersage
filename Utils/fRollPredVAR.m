function mYhat = fRollPredVAR(rModel, mY, mX)
% Function for predicting a time-series with a VAR model
%
% Input:

arguments
    rModel struct
    mY (:,:) {mustBeNumeric}
    mX (:,:) {mustBeNumeric} = []
end

% Determine dimensions
[iNumObs, iNumSeries] = size(mY);

% Lag time-series
mYlag = lagmatrix(mY,1:rModel.iNumLags);

% Replace lags with last observed in-sample observations
for iIdxL = 1:rModel.iNumLags
    % Get index of columns corresponding to lag = iIdxL
    vIdxCol = (1+iNumSeries*(iIdxL-1)):(iIdxL*iNumSeries);

    % Add data
    mYlag(1:iIdxL,vIdxCol) = rModel.mY(end-iIdxL+1:end,:);
end
    
% Get all regressors
mXreg = [ones(iNumObs, rModel.lEstAlpha), mYlag, mX];

% Prediction
mYhat = mXreg * rModel.mCoef;
end