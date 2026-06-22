% Skript zum Einlesen, Anpassen und Anwenden eines linearen Regressionsmodells
% Datensatz: kuenstliche lineare Daten

% Alles loeschen
clear; clc; close;

% Pfade setzen
sOldPath = path;
sDataPath = './DATA/Artificial/';       % Hier liegen unsere Daten
addpath('./Utils');                     % Hier liegen Hilfsfunktionen

% Daten laden
load([sDataPath, 'ArtDataLinMod.mat']);
% load([sDataPath, 'ArtDataQuadMod.mat']);

% Anzahl der Beobachtungen bestimmen
iNumObs = size(mX,1);   

% Lineare Regression schaetzen

% Variante 1a: regstats: Liefert viele Regressionsstatistiken zurueck,
% allerdings dadurch auch extrem speicherintensiv. Die benoetigten
% Statistiken koennen aber in einem Cell-Array mit Strings spezifiziert
% werden (beispielsweise interessieren hier nur die Koeffizienten, die
% t-Werte und das R2) Konstante wird automatisch ergaenzt
rLinMod = regstats(vY, mX,'linear',{'beta','tstat','rsquare'});

% Parameter auslesen
vBeta = rLinMod.beta;

% Variante 1b: regstats2: Verfuegabr unter https://de.mathworks.com/matlabcentral/fileexchange/26169-regstats2
% Selbe Funktionsweise wie bei regstats, allerdings ist die Berechnung
% robuster Standardfehler moeglich. Zudem kann eine Konstante weggelassen
% werden
if exist('regstats2','file')
    % Schaetzung mit Konstante
    rLinMod = regstats2(vY, mX,'linear',{'beta','tstat','rsquare'});

    % % Schaetzung ohne Konstante
    % rLinMod = regstats2(vY, mX,'onlydata',{'beta','tstat','rsquare'});
end

% Variante 2: regress: Schnell, liefert aber weniger Statistiken. Konstante
% muss ergaenzt werden
mXin = [ones(iNumObs,1), mX];
vBeta = regress(vY, mXin);

% Variante 3: direkte Berechnung 
mXin = [ones(iNumObs,1), mX];
vBeta = mXin\vY;

% Prognose durchfuehren (Anwendung in-sample)
mXCout = [ones(iNumObs,1), mX];        % Einser-Spalte hinzufuegen
vYest = mXCout * vBeta;                % Prognose

% Guete auswerten
[rQuality, tTable] = fForecastQuality(vY, vYest);

% Anzeige auf Konsole
fprintf('*** Guetemasse ***\n');
fprintf('MSE:          %f\n', rQuality.vMSE);
fprintf('RMSE:         %f\n', rQuality.vRMSE);
fprintf('MAE:          %f\n', rQuality.vMAE);
fprintf('MAPE:         %f\n', rQuality.vMAPE);
fprintf('Korrelation:  %f\n', rQuality.vCor);
fprintf('Trefferquote: %f\n', rQuality.vHitRate);
fprintf('Wegstrecke:   %f\n', rQuality.vProfitRate);
fprintf('Alpha:        %f\n', rQuality.rRegResults.vAlpha);
fprintf('Beta:         %f\n', rQuality.rRegResults.vBeta);
fprintf('R2:           %f\n', rQuality.rRegResults.vR2);
fprintf('p-Wert:       %f\n', rQuality.rRegResults.vP);

% Alten Pfad wiederherstellen
path(sOldPath);