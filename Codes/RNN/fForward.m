function [mYhat, mUnits, mError, mInputEC, mOutputEC] = ...
    fForward(rModel, mY, mX, vUnitInit, vErrInit)
% Forward pass in recurrent neural network
%
% Input:
%   rModel:         Struct, containing the model settings and parameters
%   mY:             N x T matrix of response variables
%                   N: number of responses
%                   T: number of time-series observations
%   mX:             L x T matrix of exogenous time-series variables
%                   L: number of time-series variables
%                   T: number of time-series observations
%   vUnitInit:      U x 1 vector of initial unit realizations
%                   U: number of units
%   vErrInit:       N x 1 vector of initial errors
%                   U: number of units
%
% Output:
%   mYhat:          N x T matrix of estimated responses
%                   N: number of responses
%                   T: number of time-series observations
%   mBeta:          N x K x T matrix of estimated betas
%                   N: number of responses
%                   K: number of output units (including intercept)
%                   T: number of time-series observations
%   mUnits:         U x T matrix of unit realizations
%                   U: number of units
%                   T: number of time-series observations
%   mError:         N x T matrux if errors
%                   N: number of responses
%                   T: number of time-series observations
%   mInputEC:       N x T matrix of inputs for error correction subnetworks
%                   N: number of responses
%                   T: number of time-series observations
%   mOutputEC:      N x H matrix of the output of the error correction
%                   network layer
%                   N: number of responses
%                   H: number of error correction units

% Determine dimensions
[iNumResp, iNumObs] = size(mY);
iNumUnits = rModel.iNumUnits;

% Set optional arguments
if nargin < 5 || isempty(vErrInit)
    vErrInit = zeros(iNumResp,1);
end
if nargin < 4 || isempty(vUnitInit)
    vUnitInit = rModel.rPrms.vUnitInit;
end

% Append bias to instruments
if rModel.rHidLayer.lEstBias 
    mX = [ones(1, iNumObs); mX];
end

% Define functions for hidden and output-layer
hActivFunHid = fGetActivFun(rModel.rHidLayer.iActivFun);
hActivFunOut = fGetActivFun(rModel.rOutLayer.iActivFun);

% Initialize memory
mUnits = NaN(iNumUnits, iNumObs);
mYhat = NaN(iNumResp, iNumObs);
mError = NaN(iNumResp, iNumObs);
mInputEC = [zeros(iNumResp, 1), NaN(iNumResp, iNumObs-1)];
mOutputEC = NaN(rModel.rErrCorrLayer.iNumUnits, iNumObs);

% Forward pass through time
for iIdxT = 1:iNumObs
    % Get initial values for unit realizations and errors
    if iIdxT == 1
        vLastUnits = vUnitInit;
        vLastError = vErrInit;
    else
        vLastUnits = mUnits(:,iIdxT-1);
        
        % Error correction network has three modes
        if rModel.rErrCorrLayer.iMode == 1
            % Ordinary error correction: Model error is input to network
            vLastError = mError(:,iIdxT-1);
        elseif rModel.rErrCorrLayer.iMode == 2
            % Last model estimate is input to network
            vLastError = mYhat(:,iIdxT-1);
        elseif rModel.rErrCorrLayer.iMode == 3
            % Last actual realization is input to network
            vLastError = mY(:,iIdxT-1);
        end
        % Replace missing values with zeros            
        vLastError(isnan(vLastError)) = 0;
    end
    
    % Get output from error correction layer
    if rModel.rErrCorrLayer.iNumUnits > 0
        % Forward pass on error correction network
        vOutputEC = fForwardErrCorrLayer(rModel, vLastError);
        mOutputEC(:,iIdxT) = vOutputEC;
        vErrorCorr = rModel.rPrms.mOmegaEC' * vOutputEC;
    else
        vErrorCorr = zeros(iNumUnits,1);
    end

    % Calculate units
    vUnitsTemp = hActivFunHid( ...
        rModel.rPrms.mOmega' * mX(:,iIdxT) + ...% Input from input layer
        rModel.rPrms.mAR' * vLastUnits + ...    % Input from previous layer
        vErrorCorr);                            % Input from error correction layer 
        
    % Append constant if required
    if rModel.rOutLayer.lEstAlpha
        vUnitsTemp = [ones(1); vUnitsTemp];
    end
    
    % Estimate response variables
    vYhatTemp = hActivFunOut( rModel.rPrms.mBeta * vUnitsTemp);
    
    % Calculate error
    if (any( rModel.rOptimOpt.iObjFun == [1, 2]))
        % Sum of squared/absolute residuals
        mError(:,iIdxT) = mY(:,iIdxT) - vYhatTemp;
    elseif (rModel.rOptimOpt.iObjFun == 3)
        % Binary cross-entropy
        dEps = 1e-10;
        mError(:,iIdxT) = -(mY(:,iIdxT) .* log(vYhatTemp + dEps) + ...
            (1 - mY(:,iIdxT)).*log(1 - vYhatTemp + dEps));
    end
    
    % Save 
    if rModel.rOutLayer.lEstAlpha
        mUnits(:,iIdxT) = vUnitsTemp(2:end);
    else
        mUnits(:,iIdxT) = vUnitsTemp;
    end
    mYhat(:,iIdxT) = vYhatTemp;
    mInputEC(:,iIdxT) = vLastError;
    
end
end
