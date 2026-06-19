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

% Reporting lag (0 for models estimated without one)
if isfield(rModel, 'iReportLag')
    iReportLag = rModel.iReportLag;
else
    iReportLag = 0;
end
iNumLags = rModel.iNumLags;

% Working series: the in-sample observations, extended with each forecast so
% that multi-step predictions feed back correctly.
mSeries = rModel.mY;
iT      = size(mSeries, 1);

% Loop over time
for iIdxT = 1:iNumOut
    % Predicting observation iT+iIdxT. With a reporting lag r and iNumLags
    % lags the regressors are y(t-(r+1)), ..., y(t-(r+iNumLags)). The indices
    % are listed most-recent-lag first to match the coefficient ordering.
    if iNumLags > 0
        vIdxLags = (iT + iIdxT) - ((iReportLag+1):(iReportLag+iNumLags));
        vLagRow  = reshape(mSeries(vIdxLags, :)', 1, []);
    else
        vLagRow  = [];
    end

    % Create matrix of predictors
    mXreg = [ones(rModel.lEstAlpha), vLagRow];

    % Add additional (exogenous) regressors if available
    if size(mX,1) >= iIdxT
        mXreg = [mXreg, mX(iIdxT,:)];
    end

    % Prediction
    vYhat = mXreg * rModel.mCoef;

    % Append the forecast so it can serve as a lag for later steps
    mSeries = [mSeries; vYhat];

    % Save prediction
    mYhat(iIdxT,:) = vYhat;
end
end