% Skript zur Generierung kuenstlicher Daten

% Alles loeschen
clear; clc; close all;

% Pfade setzen
sOldPath = path;
sDataPath = './DATA/Artificial/';   % Hier speichern wir unsere Daten
addpath('./GenerateArtData/');      % Funktionen zur Erzeugung kuenstlicher Daten

% Einstellungen fuer lineare Daten
iNumObs = 20000;            % Anzahl Beobachtungen
dConst = 1;                 % Konstante
vBeta = [1; 1; 0; 0; 0];    % Bekannte, gesetzte Koeffizienten
dNoise = 1;                 % Rauschfaktor (Standardabweichung der Residuen)
dThreshold = 1;             % Schwellenwert fuer Klassifikation in Klassen 0 und 1

% Lineare Daten generieren
[vY, vC, mX, vEps] = fGenLinData(dConst,vBeta,dNoise,dThreshold,iNumObs);

% Testdaten speichern
save([sDataPath, 'ArtDataLinMod.mat'],'vY','vC','mX','vEps');

% Ergaenzende Einstellungen fuer nicht-lineare Daten
cFunc = cell(5,1);
cFunc{1} = @(x)(x);           % Linearer Teilterm
cFunc{2} = @(x)(x.^2);        % Quadratischer Teilterm
vBetaQuad = [1; 1; 0; 0; 0]; % Koeffizienten der Teilterme/Einflussfaktoren

% Quadratische Daten generieren
[vY, vC, mX, vEps] = fGenNonLinData(dConst,cFunc,vBetaQuad,dNoise,dThreshold,iNumObs);

% Testdaten speichern
save([sDataPath, 'ArtDataQuadMod.mat'],'vY','vC','mX','vEps');

% Pfade wiederherstellen
path(sOldPath);

