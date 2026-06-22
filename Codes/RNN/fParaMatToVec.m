function [vPara, lActive] = fParaMatToVec(rModel, lGradient)
% Function for reshaping the free parameters of a recurrent neural network
%
% Input:
%   rModel:         Struct, contains the model settings and parameters
%   lGradient:      Logical, specifies whether to extract the gradient
% 
% Output:
%   vPara:          P* x 1 vector of free parameters
%                   P*: number of free parameters

% Optional arguments
if nargin < 2 || isempty(lGradient)
    lGradient = false;
end

% Choose field to extract
if lGradient
    sField = 'rGrad';
else
    sField = 'rPrms';
end

% Get parameters
vPara = [rModel.(sField).mOmega(:);...
    rModel.(sField).mTheta(:); ....
    rModel.(sField).vBiasEC;....
    rModel.(sField).mOmegaEC(:);...
    rModel.(sField).mAR(:);...
    rModel.(sField).vUnitInit(:);....
    rModel.(sField).mBeta(:)];
lActive = [rModel.rPrms.lOmega(:); ...
    rModel.rPrms.lTheta(:); ...
    rModel.rPrms.lBiasEC;...
    rModel.rPrms.lOmegaEC(:);...
    rModel.rPrms.lAR(:);...
    rModel.rPrms.lUnitInit(:);...
    rModel.rPrms.lBeta(:)];

% Keep only active (i.e., free) parameters)
vPara = vPara(logical(lActive));
end

