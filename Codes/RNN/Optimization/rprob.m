function [vParaBest,dFvalBest,vGradBest] = rprob(FUN,X,options,lb,ub)
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
%       .sDisplay:      String, specifies the display (default = 'iter')
%       .dDropoutProb:  Scalar, double, dropout probability for weights
%                       (default = 0)
%
%       .dStepPlus:
%       .dStepMinus:
%       .dMaxStep:      Scalar, double, maximum step size (default = 50)
%       .dMinStep:      Scalar, double, minimum step size (default = 1e-6)
%       .lWeightBacktracking: Logical, specifies whether to use weight
%                       backtracking (true, default) or not (false)
% Output:
%   vParaBest:      P x 1 vector of optimal parameter values
%                   P: number of parameters
%   dFvalBest:      Scalar, double objective function value
%   vGradBest:      P x 1 vector of final gradients
%                   P: number of parameters

% Set optional arguments
if nargin < 3 || isempty(options)
    options = struct();
end
% General settings
if ~isfield(options,'iMaxIter')
    options.iMaxIter = 1000;
end
if ~isfield(options,'iMaxTime')
    options.iMaxTime = Inf;
end
if ~isfield(options,'dTol')
    options.dTol = 1e-6;
end
if ~isfield(options,'iEarlyStop')
    options.iEarlyStop = 50;
end
if ~isfield(options,'sDisplay')
    options.sDisplay = 'iter';
end
if ~isfield(options,'dDropoutProb')
    options.dDropoutProb = 0;
end

% Algorithm-specific settings
if ~isfield(options,'dStepPlus')
    options.dStepPlus = 1.2;
end
if ~isfield(options,'dStepMinus')
    options.dStepMinus = 0.5;
end
if ~isfield(options,'dStepMin')
    options.dStepMin = 1e-5;
end
if ~isfield(options,'dStepMax')
    options.dStepMax = 5;
end
if ~isfield(options,'lWeightBacktracking')
    options.lWeightBacktracking = true;
end

% Number of parameters
iNumPrms = length(X);

% Initialization
dFvalBest = NaN;                    % Best objective function value
vParaBest = X;                      % Best parameter set
vPara = X;                          % Initial parameters
vGradBest = Inf(iNumPrms,1);        % Gradient of best parameter set
vLastGrad = zeros(iNumPrms,1);      % Initialization of last gradient
vLastStep = ones(iNumPrms,1) * 0.001;% Initialization of last steps
vLastUpdate = zeros(iNumPrms,1);    % Initialization of last update
dLastError = Inf;                   % Last error
iCounter = 1;
lDebug = false;
if lDebug
    vObjVal = NaN(options.iMaxIter,1);
end

tic
% Optimization
for iIdxI = 1:options.iMaxIter
    % Evaluate objective function and calculate the gradient
    [dFval, vGrad] = FUN(vPara);

    % Save objective function
    if lDebug
        vObjVal(iIdxI) = dFval;
    end
    
    % Get the step directions
    vSign = sign(vGrad .* vLastGrad);
    
    % Get the step sizes
    vStepSize = vLastStep;
    vStepSize(vSign == 1) = min(options.dStepMax, options.dStepPlus * vLastStep(vSign == 1));
    vStepSize(vSign == -1) = max(options.dStepMin, options.dStepMinus * vLastStep(vSign == -1));
    vParaCh = - sign(vGrad) .* vStepSize;
    if options.lWeightBacktracking
        % Backtraining of the weights as described in Igel/Hüksen (2000):
        % Alg. 3)
        lBackTrack = (vSign == -1); %(dFval > dLastError);
        vParaCh(lBackTrack) = -vLastUpdate(lBackTrack);
        vGrad(lBackTrack) = 0;
    end
    
    % Dropout 
    if options.dDropoutProb > 0
        % Select parameters that are muted
        lDropout = rand(iNumPrms,1) < options.dDropoutProb;
        
        % Set their updates to zero
        vParaCh(lDropout) = 0;
        vStepSize(lDropout) = vLastStep(lDropout);
    end

    
%     vParaCh = NaN(iNumPrms,1);
%     vStepSize = NaN(iNumPrms,1);
%     for iIdxP = 1:iNumPrms
%         if (vGrad(iIdxP) * vLastGrad(iIdxP)) > 0
%             vStepSize(iIdxP) = min(options.dStepMax, options.dStepPlus * vLastStep(iIdxP));
%             vParaCh(iIdxP) = - sign(vGrad(iIdxP)) * vStepSize(iIdxP);
%         elseif (vGrad(iIdxP) * vLastGrad(iIdxP)) < 0
%             vStepSize(iIdxP) = max(options.dStepMin, options.dStepMinus * vLastStep(iIdxP));
%             vParaCh(iIdxP) = -vLastUpdate(iIdxP);
%             vGrad(iIdxP) = 0;
%         else
%             vStepSize(iIdxP) = vLastStep(iIdxP);
%             vParaCh(iIdxP) = - sign(vGrad(iIdxP)) * vLastStep(iIdxP);
%         end
%     end

    % Reset parameters if they violate constraints
    lVioLB = vPara < lb;
    lVioUB = vPara > ub;
    if ~isempty(lb) && any(lVioLB)
%          warning('Resetting');
        vPara(lVioLB) = lb(lVioLB);
    end
    if ~isempty(ub) && any(lVioUB)
%          warning('Resetting');
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

    % Update weights
    vPara = vPara + vParaCh;
    vLastGrad = vGrad;
    vLastStep = vStepSize;
    dLastError = dFval;
    vLastUpdate = vParaCh;
    
    % Check if early stopping
    if  (toc >= options.iMaxTime) ||((options.iEarlyStop > 0) && (iCounter > options.iEarlyStop))
        if strcmp(options.sDisplay,'iter')
            fprintf('Optimization terminated due to early stopping \n');
            fprintf('Iterations: %i, fval: %d \n',iIdxI,dFvalBest);
        end
        break
    end
    
    % Print on console (every 50 iterations)
    iNumIter = 1;
    if strcmp(options.sDisplay,'iter')
        if (iIdxI == 1) || (floor(iIdxI/iNumIter) == (iIdxI/iNumIter))
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