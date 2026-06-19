function [dOf, vGrad] = fObjFunRNN(rModel, vPara, mY, mX, mWts)
% Function for evaluating the objective function and for calculating the
% gradient of the loss with respect to the parameters of an error correction
% recurrent neural network
%
% Input:
%   rCondNN:        Struct, object of a conditional neural network
%   vPara:          P* x 1 vector of model parameters
%                   P*: number of parameters (only active)
%   mY:             N x T matrix of response variables
%                   N: number of responses
%                   T: number of time-series observations
%   mX:             L x T matrix of exogenous time-series variables
%                   L: number of time-series variables
%                   T: number of time-series observations
%   mWts:           N x T matrix of observation weights
%                   N: number of responses
%                   T: number of time-series observations
%
% Output:
%   dOf:            Scalar, double, objective function value
%   vGrad:          P* x 1 vector of the gradient w.r.t. each free parameter
%                   P: number of parameters

% Determine dimensions
[iNumResp, iNumObs] = size(mY);
iNumUnits = rModel.iNumUnits;
    
% Update parameter matrices of model
rModel = fParaVecToMat(rModel,vPara);

% Forward pass
[mYhat, mUnits, mError, mInputEC, mOutputEC] = fForward(rModel, mY, mX);

% Objective function value
dOf = fObjFun(mError, mWts, rModel.rOptimOpt.iObjFun);

%% Backward pass
if nargout > 1
    % Define activation functions for hidden and output-layer and get their
    % derivatives
    [~,hDerivActivFunHid] = fGetActivFun(rModel.rHidLayer.iActivFun);
    [~, hDerivActivFunOut] = fGetActivFun(rModel.rOutLayer.iActivFun); 

    % Calculate derivative of loss function
    mDeriv = fDerivObjFun(mY, mYhat, mWts, rModel.rOptimOpt.iObjFun);
    mDeriv = mDeriv .* hDerivActivFunOut(mYhat);
    mDeriv(isnan(mDeriv)) = 0;
    
    % Derivative with respect to units
    mDeltaUnit = hDerivActivFunHid(mUnits);
    
    % Add constant to units if specified
    if rModel.rOutLayer.lEstAlpha
        mUnitsOut = [ones(1, iNumObs); mUnits];
    else
        mUnitsOut = mUnits;
    end
    
    % Add bias to variables
    if rModel.rHidLayer.lEstBias
        mX = [ones(1, iNumObs); mX];
    end
    
    % Initialize memory
    mErrSigHid = zeros(iNumUnits, iNumObs);         % Error signal at hidden layer
    vLastDeltaOut = zeros(iNumResp, 1);             % Error signal from error correction layer is zero at T
    vLastErrSigHid = zeros(iNumUnits,1);            % Error signal from previous layer is zero at T
    
    % Backpropagation through time
    for iIdxT = iNumObs:-1:1
        % Get error signal at output-layer (error + error of EC network)
        vDeltaOut = mDeriv(:,iIdxT) + vLastDeltaOut;
        
        % === Gradient at output-layer
        % Error signal 
        mErrSigOutUnit = (mUnitsOut(:,iIdxT) * vDeltaOut');
        
        % Gradient w.r.t beta
        rModel.rGrad.mBeta = rModel.rGrad.mBeta + mErrSigOutUnit';
        
        % === Gradient at hidden layer
        % Outer error signal
        if rModel.rOutLayer.lEstAlpha
            vDeltaHid = (rModel.rPrms.mBeta(:,2:end)' * vDeltaOut) .* mDeltaUnit(:,iIdxT);
        else
            vDeltaHid = (rModel.rPrms.mBeta' * vDeltaOut) .* mDeltaUnit(:,iIdxT);
        end
        
        % Inner error signal
        vDeltaInner = (rModel.rPrms.mAR * vLastErrSigHid) .* mDeltaUnit(:,iIdxT);
        
        % Total error signal at hidden layer
        vErrSigHid = vDeltaHid + vDeltaInner;

        % Update gradient with respect to omega
        rModel.rGrad.mOmega = rModel.rGrad.mOmega + (mX(:,iIdxT) * vErrSigHid');

        % Gradient update w.r.t. AR follows after loop
        
        % ======== ERROR CORRECTION NETWORK ===========
        if rModel.rErrCorrLayer.iNumUnits > 0
            rModel.rGrad.mOmegaEC = rModel.rGrad.mOmegaEC + ...
                (vErrSigHid * mOutputEC(:,iIdxT)')';
            [mGradW, vGradB, vLastDeltaOut] = fGradErrCorr(rModel,...
                vErrSigHid, mOutputEC(:,iIdxT), mInputEC(:,iIdxT));
            rModel.rGrad.mTheta = rModel.rGrad.mTheta + mGradW;
            rModel.rGrad.vBiasEC = rModel.rGrad.vBiasEC + vGradB;
            
            if rModel.rErrCorrLayer.iMode == 2
                % Change sign
                vLastDeltaOut = - vLastDeltaOut;
            elseif rModel.rErrCorrLayer.iMode == 3
                % Teacher enforcing
                vLastDeltaOut = zerps(iNumResp,1);
            end
        else
            vLastDeltaOut = zeros(iNumResp, 1);
        end

        % Save
        mErrSigHid(:,iIdxT) = vErrSigHid;
    end
    % Calculate gradient w.r.t. AR and initialization
    rModel.rGrad.mAR = [rModel.rPrms.vUnitInit, mUnits(:,1:end-1)] * mErrSigHid';
    rModel.rGrad.vUnitInit = rModel.rPrms.mAR * mErrSigHid(:,1);
    
    % Get gradient
    vGrad = fParaMatToVec(rModel, true);     
end
end
