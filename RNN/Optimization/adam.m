function [vParaBest,dFval,vGradBest] = adam(FUN,X,options,lb,ub)
% Funktion zur Loesung eines Optimierungsproblems ohne Restriktionen mit
% dem Adam-Algorithmus

% Optionale Argumente
if nargin < 3 || isempty(options)
    options = struct();
end
if ~isfield(options,'iMaxIter')
    options.iMaxIter = 1000;
end
if ~isfield(options,'iMaxTime')
    options.iMaxTime = Inf;
end
if ~isfield(options,'dAlpha')
    options.dAlpha = 0.1;
end
if ~isfield(options,'dBeta1')
    options.dBeta1 = 0.8;
end
if ~isfield(options,'dBeta2')
    options.dBeta2 = 0.999;
end
if ~isfield(options,'dEps')
    options.dEps = 1e-8;
end
if ~isfield(options,'iEarlyStop')
    options.iEarlyStop = 50;
end

% Dimensionen bestimmen
iNumPrms = length(X);

% Bester Zielfunktionswert
dElapsedTime = 0;
dFvalBest = NaN;
vParaBest = X;
vGradBest = Inf(iNumPrms,1);
iCounter = 0;
vPara = X;
vM = zeros(iNumPrms,1);
vV = zeros(iNumPrms,1);

% Start time measurement
tic
for iIdxI = 1:options.iMaxIter
    % Zielfunktion bei ggb. Parametern auswerten
    [dFval, vGrad] = FUN(vPara);
    
    % Parameter updaten
    vM = options.dBeta1 * vM + (1 - options.dBeta1) * vGrad;
    vV = options.dBeta2 * vV + (1 - options.dBeta2) * vGrad.^2;
    
    % Bias-Korrektur
    vMhat = vM/(1 - options.dBeta1.^(iIdxI+1));
    vVhat = vV/(1 - options.dBeta2.^(iIdxI+1));
    
    % Beste Zielfunktionswert speichern
    if (dFvalBest > dFval) || isnan(dFvalBest)
        % Beste Loesung speichern
        dFvalBest = dFval;
        vParaBest = vPara;
        vGradBest = vGrad;
        
        % Wenn eine bessere Loesung gefunden wurde, dann hier den Counter 
        % auf null setzen
        iCounter = 0;
    else
        % Keine bessere Loesung gefunden, also Counter erhoehen
        iCounter = iCounter + 1;
    end
    
    % Pruefen, ob early stopping zur Anwendung kommt
    if ((options.iEarlyStop > 0) && (iCounter > options.iEarlyStop)) || (toc >= options.iMaxTime)
        break
    end
    
    % Gewichte updaten
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
end
end

