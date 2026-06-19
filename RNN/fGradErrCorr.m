function [mGradTheta, vGradBias, vLastDelta] = fGradErrCorr(rModel,vErrSig,vOutputEC,vErrorIn)
% Function for calculating the gradient of the inner error correction
% network
%
% Input:
%   rCondNN:            Struct, object, conditional neural network
%   vErrSig:            U x 1 vector, error signal at hidden layer
%                       U: number of units (output + hidden)
%   vOutputEC:          H x 1 vector of outputs of error hidden units
% vErrorIn:     N x 1 vector of errors


% Rueckwaertspropagierung des Fehlersignals
vDelta = rModel.rPrms.mOmegaEC * vErrSig;

% Get derivative
[~,hDerivActivFun] = fGetActivFun(rModel.rErrCorrLayer.iActivFun);
vErrSigHidErr = hDerivActivFun(vOutputEC) .* vDelta;

% Gradientenberechnung Verbindungen Error-zu-HiddenError-Layer
mGradTheta = vErrorIn * vErrSigHidErr';
% mGradTheta = mGradTheta';

% Gradientenberechnung am Bias des HiddenError-Layer
vGradBias = vErrSigHidErr;

% Rueckpflanzung des ErrorCorrection Teil, Vorzeichen umkehren nicht
% vergessen
vLastDelta = -(rModel.rPrms.mTheta * vErrSigHidErr);
end

