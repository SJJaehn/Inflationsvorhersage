function rModel = fOptRNN(rModel,mY, mX, mWts)
% Function for optimizing the free parameters of a recurrent neural network
%
% Input:
%   rModel:         Struct, containing the model settings and parameters
%                   following fields
%   mY:             N x T matrix of the response variable
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
%   rModel:         Struct, trained model

%% Prepare input (i.e., dropout and overshooting)
% Overshooting: Remove the last observations
mX(:,end-rModel.rHidLayer.iNumOvershoot+1:end) = 0;

% Dropout: Set random observations to missing
lDrop = rand(1,size(mY,2)) <= rModel.rHidLayer.dDropout;
mX(:,lDrop) = 0;

% Get the initial values for the free parameters of the model
vPara0 = fParaMatToVec(rModel);

% Define the objective function 
hObjFun = @(vPara)fObjFunRNN(rModel, vPara, mY, mX, mWts);

% Debugging: Get initial objective function value
dObjValInit = hObjFun(vPara0);

%% Optimization
% Set options
rOptimOpt.iMaxIter = rModel.rOptimOpt.iMaxIter;     % Maximum number of iterations
rOptimOpt.sDisplay = rModel.rOptimOpt.sDisplay;    % Display progress
rOptimOpt.iMaxTime = rModel.rOptimOpt.iMaxTime;    % Time limit

% Determine number of parameters to estimate
iNumPrms = length(vPara0);

% Set upper and lower boundaries
vLB = ones(iNumPrms,1) * rModel.rOptimOpt.vLB;
vUB = ones(iNumPrms,1) * rModel.rOptimOpt.vUB;

% Optimization
if strcmp(rModel.rOptimOpt.sSolver,'adam')
    % === Optimization with ADAM
    [vParaEst,dObjVal,vGrad] = ...
        adam(@(vPara)hObjFun(vPara),vPara0,rOptimOpt,vLB,vUB);

elseif strcmp(rModel.rOptimOpt.sSolver,'nadam')
    % === Optimization with NADAM
    [vParaEst,dObjVal,vGrad] = ...
        nadam(@(vPara)hObjFun(vPara),vPara0,rOptimOpt,vLB,vUB);

elseif strcmp(rModel.rOptimOpt.sSolver,'rmsprob')
    % === Optimization with RMSprob
    [vParaEst,dObjVal,vGrad] = ...
        rmsprob(@(vPara)hObjFun(vPara),vPara0,rOptimOpt,vLB,vUB);

elseif strcmp(rModel.rOptimOpt.sSolver,'rprob')
    % === Optimization with RMSprob
    [vParaEst,dObjVal,vGrad] = ...
        rprob(@(vPara)hObjFun(vPara),vPara0,rOptimOpt,vLB,vUB);
end

% Get optimal model
rModel = fParaVecToMat(rModel, vParaEst);
rModel.dObjValInit = dObjValInit;
rModel.dObjVal = dObjVal;

% Forward pass
[rModel.mYhat, rModel.mUnits, rModel.mError, rModel.mInputEC, rModel.mOutputEC] = ...
    fForward(rModel, mY, mX);

end