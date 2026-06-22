function rModel = fSetupRNN(rModel)
% Function for setting default options

%% Check inputs
if nargin < 1 || isempty(rModel)
    rModel = struct();
else
    assert(isstruct(rModel),'CondNN object must be a struct');
end

%% Check general network architecture
% Number of units, if no value is passed, the code should throw an error
rModel = fCheckField(rModel, 'iNumUnits', NaN);

%% Set up the hidden layer
% Ensure that field exists
rModel = fCheckField(rModel, 'rHidLayer', struct());

% Specify whether to estimate a bias, default = true 
rModel.rHidLayer = fCheckField(rModel.rHidLayer, 'lEstBias', true);

% Specify which activation function to use at hidden layer
% 1: linear
% 2: sigmoid
% 3: tanh (default)
% 4: relu
rModel.rHidLayer = fCheckField(rModel.rHidLayer, 'iActivFun', 3);

% Specify whether to activate dropout of time series, default = 0
rModel.rHidLayer = fCheckField(rModel.rHidLayer, 'dDropout', 0);

% Specify whether to activate overshooting, default = 0
rModel.rHidLayer = fCheckField(rModel.rHidLayer, 'iNumOvershoot', 0);

%% Set up the error correction network
% Ensure that field exists
rModel = fCheckField(rModel, 'rErrCorrLayer', struct());

% Specify number of error correction units
rModel.rErrCorrLayer = fCheckField(rModel.rErrCorrLayer, 'iNumUnits', 0);

% Specify the error correction type if activated, default = 1 = classical
% error correction
rModel.rErrCorrLayer = fCheckField(rModel.rErrCorrLayer, 'iMode', 1);

% Specify the activation function
% 1: linear (default)
% 2: sigmoid
% 3: tanh
% 4: relu
rModel.rErrCorrLayer = fCheckField(rModel.rErrCorrLayer, 'iActivFun', 3);

%% Specify output layer
% Ensure that field exists
rModel = fCheckField(rModel, 'rOutLayer', struct());

% Specify whether to estimate intercept/alpha, default = false
rModel.rOutLayer = fCheckField(rModel.rOutLayer, 'lEstAlpha', false);

% Specify the activation function for responses
% 1: linear (default)
% 2: sigmoid
% 3: tanh
% 4: relu
rModel.rOutLayer = fCheckField(rModel.rOutLayer, 'iActivFun', 1);

%% Optimization settings
% Ensure that field exists
rModel = fCheckField(rModel, 'rOptimOpt', struct());

% Specify objective function
% 1: sum of squared residuals
% 2: sum of absolute residuals
% 3: cross-entropy
rModel.rOptimOpt = fCheckField(rModel.rOptimOpt, 'iObjFun', 1);

% Specify solver
rModel.rOptimOpt = fCheckField(rModel.rOptimOpt, 'sSolver', 'rprob');

% Specify whether to display the progress 
rModel.rOptimOpt = fCheckField(rModel.rOptimOpt, 'sDisplay', 'iter');

% Specify the maximum number of iterations for optimization
rModel.rOptimOpt = fCheckField(rModel.rOptimOpt, 'iMaxIter', 500);

% Specify the maximum time in seconds for optimization
rModel.rOptimOpt = fCheckField(rModel.rOptimOpt, 'iMaxTime', Inf);

% Specify the upper and lower bounds
rModel.rOptimOpt = fCheckField(rModel.rOptimOpt, 'vLB', -100);
rModel.rOptimOpt = fCheckField(rModel.rOptimOpt, 'vUB', 100);
end