function [vParaBest,dFvalBest,vGradBest] = nadam(FUN,X,options,lb,ub)
% Function for solving an optimization problem with nadam of Dozat (2016)
%
% Input:
%   FUN:            Objective function, returning the objective function
%                   value and the gradient
%   X:              P x 1 vector of parameters
%                   P: number of parameters
%   options:        Struct, containing some options
%       .iMaxIter:      Scalar, maximum number of iterations
%                       (default = 1000)
%       .iEarlyStop:    Scalar, integer, number of tolerated iterations
%                       where no improvement in the objective function
%                       value was recorded. Exit after limit is reached
%                       (default = 50)
%       .dAlpha:        Scalar, double, initial learning rate (default =
%                       0.01)
%       .dBeta1:        Scalar, double, exponential decay rate for the
%                       first moment estimates (default = 0.9)
%       .dBeta2:        Scalar, double exponential decay rate for the 
%                       second moment estimates (default = 0.999)
%       .dEps:          Scalar, double, small number of prevent any
%                       division by zero (default = 1e-10)
%   lb:             P x 1 vector of lower boundaries
%   ub:             P x 1 vector of upper boundaries
%
% Output:
%   vParaBest:      P x 1 vector of optimal parameter values
%                   P: number of parameters
%   dFvalBest:      Scalar, double objective function value
%   vGradBest:      P x 1 vector of final gradients
%                   P: number of parameters

% Set optional arguments
if nargin < 5 || isempty(ub)
    ub = [];
end
if nargin < 5 || isempty(lb)
    lb = [];
end
if nargin < 3 || isempty(options)
    options = struct();
end
if ~isfield(options,'iMaxIter')
    options.iMaxIter = 1000;
end
if ~isfield(options,'dAlpha')
    options.dAlpha = 0.001;
end
if ~isfield(options,'dBeta1')
    options.dBeta1 = 0.975;
end
if ~isfield(options,'dBeta2')
    options.dBeta2 = 0.999;
end
if ~isfield(options,'dEps')
    options.dEps = 1e-10;
end
if ~isfield(options,'iEarlyStop')
    options.iEarlyStop = 50;
end
if ~isfield(options,'sDisplay')
    options.sDisplay = 'iter';
end
if ~isfield(options,'dTol')
    options.dTol = 1e-6;
end

% Number of parameters
iNumPrms = length(X);

% Initialization
dFvalBest = NaN;                % Best objective function value
vParaBest = X;                  % Best parameter set
vGradBest = Inf(iNumPrms,1);    % Gradient of best parameter set
iCounter = 0;                   % Number of iterations where fval did not improve
vPara = X;                      % Initial parameters
vM = zeros(iNumPrms,1);         % Initial estimate for first moment
vV = zeros(iNumPrms,1);         % Initial estimate for second moment
for iIdxI = 1:options.iMaxIter
    % Eval objective function
    [dFval, vGrad] = FUN(vPara);
    
    % Update parameters
    vM = options.dBeta1 * vM + (1 - options.dBeta1) * vGrad;
    vV = options.dBeta2 * vV + (1 - options.dBeta2) * vGrad.^2;
    
    % Bias correction
    vMhat = (options.dBeta1 * vM/(1-options.dBeta1)) + ...
        ((1 - options.dBeta1) * vGrad/(1 - options.dBeta1));
    vVhat = options.dBeta2 * vV/(1 - options.dBeta2);
    
    % Store best objective function value
    if (dFvalBest > dFval) || isnan(dFvalBest)
        % Save best solution
        dStepSize = dFvalBest - dFval;
        dFvalBest = dFval;
        vParaBest = vPara;
        vGradBest = vGrad;
        
        % Set counter to zero if best solution was found
        if dStepSize > options.dTol
            iCounter = 0;
        else
            iCounter = iCounter + 1;
        end
    else
        % Increase counter if no better solution was found
        iCounter = iCounter + 1;
    end
    
    % Check if early stopping
    if (options.iEarlyStop > 0) && (iCounter > options.iEarlyStop)  && (strcmp(options.sDisplay,'iter'))
        fprintf('Optimization terminated due to early stopping \n');
        fprintf('Iterations: %i, fval: %d \n',iIdxI,dFvalBest);
        break
    end
    
    % Update weights
    vPara = vPara - options.dAlpha * (vMhat./(sqrt(vVhat) + options.dEps));

    % Reset parameters if the violate constraints
    lVioLB = vPara < lb;
    lVioUB = vPara > ub;
    if ~isempty(lb) && any(lVioLB)
        warning('Resetting');
        vPara(lVioLB) = lb(lVioLB);
    end
    if ~isempty(ub) && any(lVioUB)
        warning('Resetting');
        vPara(lVioUB) = ub(lVioUB);
    end
    
    % Print on console (every 50 iterations)
    if strcmp(options.sDisplay,'iter')
        if (iIdxI == 1) || (floor(iIdxI/50) == (iIdxI/50))
            iNumIt = length(num2str(iIdxI));
            iNumFval = length(num2str(dFvalBest));
            iNumStep = length(num2str(dStepSize));
            if iIdxI == 1
                fprintf('Iter      | Best         | Step Size    \n');
                fprintf('----------|--------------|------------- \n');
            end
            fprintf(['%i',repmat(' ',1,10-iNumIt),'| %d',repmat(' ',1,10-iNumFval),...
                '%d',repmat(' ',1,10-iNumFval),'\n'], iIdxI, dFvalBest,dStepSize);
        end
    end
end

if strcmp(options.sDisplay,'iter') && (iIdxI == options.iMaxIter)
    fprintf('Iteration limit reached. fval: %d \n',dFvalBest);
end
        