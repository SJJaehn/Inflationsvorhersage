function vErrorHidOut = fForwardErrCorrLayer(rModel,vLastError)
% Function for implementing the forward pass of the error correction
% network layer
%
% Input:
%   rCondNN:        Struct, conditional neural network
%   vLastError:     N x 1 vector of errors of previous time step
%
% Output
%   vErrorHidOut:   H x 1 vector, output of error correction network layer

% Obtain input at error correction layer
vInput = rModel.rPrms.mTheta' * vLastError + rModel.rPrms.vBiasEC;

% Implemented activation functions (1: linear, 2: sigmoid, 3: tanh, 4: relu)
hActivFun = fGetActivFun(rModel.rErrCorrLayer.iActivFun);
vErrorHidOut = hActivFun(vInput);
end

