function mYhat = fPredictVAR(rModel, iNumOut, mX)
% Function for predicting using a VAR Model

% Check input arguments
arguments
    rModel
    iNumOut (1,1) {mustBeNumeric, mustBeNonnegative} = 1
    mX (:,:) {mustBeNumeric} = []
end

% Initialize memory
mYhat = NaN(iNumOut, rModel.iNumDepVars);

% Get last Y values
mYlast = rModel.mY(end-rModel.iNumLags+1:end,:);

% The coefficients are sorted such that the first coefficients correspond
% to the most recent lags. So we need to flip the last observed values. Now
% most recent value is in the top row and oldest value in the bottom row
mYlast = flipud(mYlast);

% Loop over time
for iIdxT = 1:iNumOut
    % Create matrix of predictors
    mXreg = [ones(rModel.lEstAlpha), reshape(mYlast',1,[])];

    % Add additional predictions if available
    if size(mX,1) >= iIdxT
        mXreg = [mXreg, mX(iIdxT,:)];
    end

    % Prediction
    vYhat = mXreg * rModel.mCoef;

    % Now drop oldest values and add predictions as most recent observations
    mYlast = [vYhat; mYlast(1:end-1,:)]; 

    % Save prediction
    mYhat(iIdxT,:) = vYhat;
end
end