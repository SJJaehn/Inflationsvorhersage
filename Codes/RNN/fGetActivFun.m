function [hActivFun,hDerivActivFun] = fGetActivFun(iWhichFun)
% Function for returning the activation function and its derivative
% 
% Input:
%   iWhichFun:          Scalar, integer, specifies the activation function
%                       1: linear
%                       2: sigmoid
%                       3: tanh
%                       4: relu
%
% Output:
%   hActivFun:          Function handle, activation function
%   hDerivActivFun:     Function handle, derivative of activation function

% Specify activation function
hActivFun = {@(x)x; ...
    @(x)(1./(1 + exp(-x))); ...
    @(x)tanh(x); ...
    @(x)max(0,x)};
hActivFun = hActivFun{iWhichFun};

% Specify derivative of activation function
hDerivActivFun = {@(x)ones(size(x)); ...
    @(x)(x.*(1 - x)); ...
    @(x)(1 - (x).^2);...
    @(x)double(x > 0);};
hDerivActivFun = hDerivActivFun{iWhichFun};
end

