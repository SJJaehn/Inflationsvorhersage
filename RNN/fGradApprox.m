function [ vGrad, dFx, vFxPh, vFxMh, vHi ] = ...
    fGradApprox( hFunc, vX, dEps, iOption, varargin )
% fGradApprox.m
% Funktion zur Approximation des Gradienten einer Funktion mit skalarem
% Output (Regelfall). Hinweis: Die Funktion kann auch den Gradienten einer
% Funktion mit vektoriellem Output approximieren (Ausnahmefall).
%
% Aufruf:
%   [ vGrad, dFx, vFxPh, vFxMh, vHi ] = ...
%       fGradApprox( hFunc, vX, dEps, iOption, varargin )
%
% Input:
%   hFunc:    function handle auf Zielfunktion, ersatzweise String mit 
%             Namen der Funktion
%   vX:       Kx1 Vektor der Argumente der Zielfunktion.
%   dEps:     Skalar, Schrittweite
%   iOption:  Art des Gradienten
%             1: rechtsseitig (default)
%             2: linksseitig
%             3: zentral
%   varargin: Cell-Array mit weiteren zu uebergebenden Inputparametern 
%             fuer die Zielfunktion 
%
% Output:
%   vGrad:    K x 1 Vektor der ersten partiellen Ableitungen,
%             Approximation des Gradienten.  Wenn die Funktion
%             hFunc einen Mx1 vektoriellen Output liefert, so ist vGrad
%             ausnahmsweise eine Matrix der Dimension M x K.
%   dFx:      Skalar, Funktionswert an der Stelle vX. Wenn die Funktion
%             hFunc einen Mx1 vektoriellen Output liefert, so ist dFx
%             ausnahmsweise ein Vektor der Dimension M x 1.
%   vFxPh:    K x 1 Vektor der Funktionswerte an den Stellen f(x + h). 
%             Wenn die Funktion hFunc einen Mx1 vektoriellen Output 
%             liefert, so ist vFxPh ausnahmsweise eine Matrix der 
%             Dimension M x K.
%   vFxMh:    K x 1 Vektor der Funktionswerte an den Stellen f(x - h),
%             nur bei iOption = 3, sonst NaN.
%             Wenn die Funktion hFunc einen Mx1 vektoriellen Output 
%             liefert, so ist vFxMh ausnahmsweise eine Matrix der 
%             Dimension M x K.
%   vHi       K x 1 Vektor der absoluten Schrittweiten h(i)
%
% Version: Februar 2014
%
% Copyright (C) 2015 Th. Poddig, A. Varmaz, Ch. Fieberg
%
% Bestandteil des Buchs: 
% Computational Finance von Th. Poddig, A. Varmaz, Ch. Fieberg
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

% Optionale Argumente pruefen
if nargin < 4 || isempty(iOption)
    iOption = 1;
end
if nargin < 3 || isempty(dEps)
    dEps = 1e-8;
end
if iOption == 2
    dEps = -dEps;
end

% Funktionswert am Punkt vX
dFx = feval(hFunc,vX,varargin{:});
dFxBase = dFx;

% Dimension Argument ermitteln
iNumPara = length(vX);

% Dimension Output ermitteln, ob Sonderfall vorliegt
iNumOut = length(dFx);

% Speicherallokation
if iNumOut == 1
    % Normalfall
    vGrad = NaN(iNumPara,1);
    vFxPh = NaN(iNumPara,1);
    vFxMh = NaN(iNumPara,1);
    vHi   = NaN(iNumPara,1);
else
    % Sonderfall
    vGrad = NaN(iNumOut,iNumPara);
    vFxPh = NaN(iNumOut,iNumPara);
    vFxMh = NaN(iNumOut,iNumPara);
    vHi   = NaN(iNumOut,iNumPara);
end
    
% Funktionswert plus Schrittweite sowie Gradient berechnen
for iI=1:iNumPara
    vXi = vX;                % Stelle vX kopieren
    % --- Schrittweite hi ausrechnen
    % Absolutbetrag xi berechnen
    dAXi = abs(vX(iI));
    % epsilon (+ = rechtsseitig, - = linksseitig) mit Absolutbetrag
    % von xi multiplizieren, aber zu kleine numerische Werte verhindern
    dHi = dEps * max(dAXi,0.01);
    % --- Statement fuer Kontrolle/Kompatibilitaet mit Buch SOO, S. 684
    % Nur fuer Vergleich mit den Codes in SOO aktivieren
    %dHi = abs(vX(iI)) * 0.00001 + 1e-30;
    % ----
    vHi(iI) = dHi;           % absolute Schrittweite speichern
    vXi(iI) = vX(iI) + dHi;  % Neue Stelle vX + hi berechnen
    % Funktion an neuer Stelle auswerten
    dFxPlusH = feval(hFunc,vXi,varargin{:});
    % Funktionswert speichern
    if iNumOut == 1
        vFxPh(iI) = dFxPlusH;
    else
        vFxPh(:,iI) = dFxPlusH;
    end
    % Wenn zentraler Differenzenquotient, vX - hi berechnen
    if iOption == 3
        vXi(iI) = vX(iI) - dHi; % Schritt rueckwaerts
        % Funktionswert berechnen
        dFxMinusH = feval(hFunc,vXi,varargin{:});
        % Funktionswert speichern
        if iNumOut == 1
            vFxMh(iI) = dFxMinusH;
        else
            vFxMh(:,iI) = dFxMinusH;
        end
        dFxBase = dFxMinusH; % Basisfunktionswert anpassen
        dHi = 2 * dHi;       % Achtung: doppelte Schrittweite!
    end
    % Differenzenquotient berechnen
    dGrad = (dFxPlusH - dFxBase) / dHi;
    % Differenzenquotient speichern
    if iNumOut == 1
        vGrad(iI) = dGrad;
    else
        vGrad(:,iI) = dGrad;
    end
end

end