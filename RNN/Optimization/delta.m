function [vParaBest,dFvalBest,vGradBest] = delta(FUN,X,options,lb,ub)
% Function for solving an optimization problem with Rprob 
%
% Input:
%   FUN:            Objective function, returning the objective function
%                   value and the gradient
%   X:              P x 1 vector of parameters
%                   P: number of parameters
%   options:        Struct, containing some options
%       .iMaxIter:      Scalar, maximum number of iterations
%                       (default = 1000)
%       .dTol:          Scalar, double, optimization tolerance 
%                       (default = 1e-6)
%       .iEarlyStop:    Scalar, integer, number of tolerated iterations
%                       where no improvement in the objective function
%                       value was recorded. Exit after limit is reached
%                       (default = 50)
%       .dAlpha:        Scalar, double, learning rate (default = 0.01),
%                       if NaN a line search algorithm is applied
%       .dMomentum:     Scalar, double, momentum (default = 0.7),
%       .sDisplay:      String, specifies the display (default = 'iter')
%       .dMaxNormGrad:  Scalar, double, maximum norm of gradient for
%                       gradient clipping (default = NaN = no clipping)
%       .dDropoutProb:  Scalar, double, dropout probability for weights
%                       (default = 0)

% Number of parameters
iNumPrms = length(X);

% Set optional arguments
if nargin < 5 || isempty(ub)
    ub = Inf;
end
if nargin < 4 || isempty(lb)
    lb = -Inf;
end
if nargin < 3 || isempty(options)
    options = struct();
end
if ~isfield(options,'iMaxIter')
    options.iMaxIter = 1000;
end
if ~isfield(options,'dTol')
    options.dTol = 1e-6;
end
if ~isfield(options,'iEarlyStop')
    options.iEarlyStop = 50;
end
if ~isfield(options,'dAlpha')
    options.dAlpha = 0.01;
end
if ~isfield(options,'dMomentum')
    options.dMomentum = 0.7;
end
if ~isfield(options,'sDisplay')
    options.sDisplay = 'iter';
end
if ~isfield(options,'dMaxNormGrad')
    options.dMaxNormGrad = NaN;
end
if ~isfield(options,'dDropoutProb')
    options.dDropoutProb = 0;
end

% Increase the size of the boundaries
if length(ub) == 1
    ub = ones(iNumPrms,1) * ub;
elseif ((length(ub) > 1) && (length(ub) < iNumPrms))
    error('Size of upper bounds must be 1 or equal to number of parameters');
end
if length(lb) == 1
    lb = ones(iNumPrms,1) * lb;
elseif ((length(lb) > 1) && (length(lb) < iNumPrms))
    error('Size of lower bounds must be 1 or equal to number of parameters');
end

% Settings for line search optimization
if isnan(options.dAlpha)
    lUseLineSearch = true;
    rOptions.dLinMinPrec = 1e-4;
    rOptions.iLinMinIter = 5;
    rOptions.dInitLambda = 0.01;
else
    lUseLineSearch = false;
end

% Initialization
dFvalBest = NaN;                    % Best objective function value
vParaBest = X;                      % Best parameter set
vPara = X;                          % Initial parameters
vGradBest = Inf(iNumPrms,1);        % Gradient of best parameter set
vLastMom = zeros(iNumPrms,1);       % Initial momentum

% Optimization
for iIdxI = 1:options.iMaxIter
    % Evaluate the objective function and calculate the gradient
    [dFval, vGrad] = FUN(vPara);
    
    % Calculate gradient with momentum
    vGradMom = options.dLearnMom * vLastMom + vGrad;
    
    % Find weight changes
    if lUseLineSearch
        % Find optimal learning rate using line search

        % Call line search
        options.dAlpha = fNLlinmin_new(FUN, vLastPara, vGradMom, rOptions);
    end
    
    % Clip gradient by norm
    if ~isnan(options.dMaxNormGrad)
        % Calculate norm
        dNorm = sum(vGradMom.^2)/iNumPrms;

        % Rescale gradient
        vGradMom = options.dMaxNormGrad * (vGradMom/dNorm);
    end
    
    % Dropout 
    if options.dDropoutProb > 0
        % Select parameters that are muted
        lDropout = rand(iNumPrms,1) < options.dDropoutProb;
        
        % Set their updates to zero
        vGradMom(lDropout) = 0;
    end
    
    % Update parameters
    vPara = vPara - options.dAlpha * vGradMom;
    vLastMom = vGradMom;
    
    % Reset parameters if they violate constraints
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
end
    