function [rQuality, tTable] = fForecastQualityClasses(mCtrue,mCest)
% Funktion zur Auswertung der Prognoseguete bei zwei Klassen
%
% Input:
%
%   mCtrue:  T x L Matrix der abhaengigen Variablen, wahre Werte
%   mCest:   T x L Matrix der abhaengigen Variablen, geschaetzte Werte
%
% Output:
%   rQuality: struct, Record, mit den Auswertungen, in den Feldern:
%    vNumObsClass0true: 1 x L Vektor mit Anzahl der wahren Beobachtungen 
%                       der Klasse 0
%    vNumObsClass1true: 1 x L Vektor mit Anzahl der wahren Beobachtungen
%                       der Klasse 1
%    vNumObsClass0est:  1 x L Vektor mit Anzahl der geschaetzten
%                       Zugehoerigkeiten zu Klasse 0
%    vNumObsClass1est:  1 x L Vektor mit Anzahl der geschaetzten
%                       Zugehoerigkeiten zu Klasse 1
%    mConfusion:        3 x 3 Konfusionsmatrix mit
%     (1,1)             wahre Klasse 0 und auch als Klasse 0 geschaetzt
%     (2,1)             wahre Klasse 1 aber als Klasse 0 geschaetzt
%     (1,2)             wahre Klasse 0 aber als Klasse 1 geschaetzt
%     (2,2)             wahre Klasse 1 und auch als Klasse 1 geschaetzt
%     (1,3)             Zeilensumme 1 (wahre Klasse 0)
%     (2,3)             Zeilensumme 2 (wahre Klasse 1)
%     (3,1)             Spaltensumme 1 (geschaetzte Klasse 0)
%     (3,2)             Spaltensumme 2 (geschaetzte Klasse 1)
%     (3,3)             Totale Summme
%    vHitRate:          1 x L Vektor mit den Trefferraten
%    vMisRate:          1 x L Vektor mit den Fehlerquoten

% Moegliche Fehler abfangen
[iRowsT,iColsT] = size(mCtrue);
[iRowsE,iColsE] = size(mCest);
if iRowsT ~= iRowsE || iColsT ~= iColsE
    error('Dimensionen der Inputmatrizen inkonsistent!');
end
if iRowsT == 1 && iColsT > 1
    fprintf('WARNUNG: Zeilenvektor uebergeben, aber Spaltenvektor erwartet!');
    fprintf('Input wird transponiert, bitte Code pruefen!');
    mCtrue = mCtrue';
    mCest = mCest';
elseif iRowsT < iColsT
    fprintf('WARNUNG: Weniger Beobachtungen als Variablen, Dimensionen vertauscht?');
    fprintf('Bitte Code pruefen, Ergebnisse sind moeglicherweise falsch!');
end

% Anzahl Beobachtungen und Variablen bestimmen
[iNumObs,iNumVars] = size(mCtrue);

% Anzahl Klassen bestimmen
vClasses = unique(mCtrue(:));
if length(vClasses) ~= 2
    error('Two classes required!');
end

% Klassen extrahieren
iClass0 = vClasses(1);
iClass1 = vClasses(2);
rQuality.iClass0 = iClass0;
rQuality.iClass1 = iClass1;

% Beobachtungen zaehlen
vNumObsClass0true = sum(mCtrue == iClass0);
vNumObsClass1true = sum(mCtrue == iClass1);
vNumObsClass0est = sum(mCest == iClass0);
vNumObsClass1est = sum(mCest == iClass1);
rQuality.vNumObsClass0true = vNumObsClass0true;
rQuality.vNumObsClass1true = vNumObsClass1true;
rQuality.vNumObsClass0est = vNumObsClass0est;
rQuality.vNumObsClass1est = vNumObsClass1est;

% Confusion Matrix erstellen
cConfusionMat = cell(1,iNumVars);
vHitRate = NaN(1,iNumVars);
vMisRate = NaN(1,iNumVars);
vPrecision = NaN(1,iNumVars);
vRecall = NaN(1,iNumVars);
vFScore = NaN(1,iNumVars);
for iIdxV=1:iNumVars
    mConfusion = NaN(3,3);
    mConfusion(1,1) = sum(mCtrue(:,iIdxV) == iClass0 & mCest(:,iIdxV) == iClass0);
    mConfusion(2,1) = sum(mCtrue(:,iIdxV) == iClass1 & mCest(:,iIdxV) == iClass0);
    mConfusion(1,2) = sum(mCtrue(:,iIdxV) == iClass0 & mCest(:,iIdxV) == iClass1);
    mConfusion(2,2) = sum(mCtrue(:,iIdxV) == iClass1 & mCest(:,iIdxV) == iClass1);
    mConfusion(1,3) = mConfusion(1,1) + mConfusion(1,2);
    mConfusion(2,3) = mConfusion(2,1) + mConfusion(2,2);
    mConfusion(3,1) = mConfusion(1,1) + mConfusion(2,1);
    mConfusion(3,2) = mConfusion(1,2) + mConfusion(2,2);
    mConfusion(3,3) = mConfusion(1,3) + mConfusion(2,3);
    cConfusionMat{iIdxV} = mConfusion;
    
    % Trefferquote und Fehlerquote bestimmen
    vHitRate(iIdxV) = (mConfusion(1,1) + mConfusion(2,2)) / iNumObs; 
    vMisRate(iIdxV) = (mConfusion(1,2) + mConfusion(2,1)) / iNumObs; 
    
    % F1-Score berechnen
    
    % Precision (Anteil aller wahren Positiven an allen Positiven
    % Vorhersagen)
    vPrecision(iIdxV) = mConfusion(2,2)/mConfusion(3,2);
    
    % Recall (Anteil aller wahren positiven an allen positiven im
    % Datensatz)
    vRecall(iIdxV) = mConfusion(2,2)/mConfusion(2,3);
    
    % F-Score berechnen
    vFScore(iIdxV) = (2*(vPrecision(iIdxV).*vRecall(iIdxV)))./(vPrecision(iIdxV) + vRecall(iIdxV));    
end
rQuality.cConfusionMat = cConfusionMat;
rQuality.vHitRate = vHitRate;
rQuality.vMisRate = vMisRate;
rQuality.vPrecision = vPrecision;
rQuality.vRecall = vRecall;
rQuality.vFScore = vFScore;

% Daten fuer Tabelle zusammenstellen, aktuell nur fuer erste Variable
if nargout > 1
    ForecastVar = [cConfusionMat{1}; ...
        [vHitRate(1), NaN(1,2)]; ...
        [vMisRate(1), NaN(1,2)]];
    cRowNames = {'True 0';'True 1';'Sum'; ...
        'Hit rate'; 'Error rate'};
    cVarNames = {'Est_0','Est_1','Sum'};
    tTable = array2table(ForecastVar,'RowNames',cRowNames,'VariableNames',cVarNames);
end

end

