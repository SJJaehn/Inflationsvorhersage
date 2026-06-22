function [rQuality, tTable] = fForecastQuality(mYtrue,mYest)
% Funktion zur Auswertung der Prognoseguete
%
% Input:
%
%   mYest:   T x L Matrix der abhaengigen Variablen, geschaetzte Werte
%   mYtrue:  T x L Matrix der unabhaengigen Variablen, wahre Werte
%
% Output:
%   rQuality: struct, Record, mit den Auswertungen, in den Feldern:
%    .vMSE:     1 x L Vektor der Mean-Squared Errors (true - est)
%    .vRMSE:    1 x L Vektor der Rooted-Mean-Squared-Errors (true - est)
%    .vMAE:     1 x L Vektor der Mean-Absolute-Errors (true - est)
%    .vMAPE:    1 x L Vektor der Mean-Absolute-Percentage-Errors (true -
%               est)
%    .vCor:     1 x L Vektor der Korrelationen (true,est)
%    .vHitRate: 1 x L Vektor der richtigen Vorzeichen
%    .ProfitRate: 1 x L Vektor der Wegstrecken
%    .rRegResults: record, struct, mit den Regressionsergebnissen von
%               wahren auf geschaetzte Werte
%      .vAlpha: 1 x L Vektor der Regressions-Alphas
%      .vBeta:  1 x L Vektor der Regressions-Betas
%      .vR2:    1 x L Vektor der R2
%      .vF:     1 x L Vektor der F-Statistiken
%      .vP:     1 x L Vektor der P-Werte
%      .vErrVar: 1 x L Vektor der geschaetzten Fehlervarianz
%   tTable: Oben aufgefuehrte Ergebnisse zusammengestellt in einem
%           Tabellenobjekt

% Anzahl Beobachtungen und Variablen bestimmen
[iNumObs,iNumVars] = size(mYtrue);

% Differenzen berechnen
mDiffMat = mYtrue - mYest;

% Absolute Differenzen berechnen
mAbsDiffMat = abs(mDiffMat);

% Differenzen quadrieren
mDiffMat2 = mDiffMat.^2;

% MSE berechnen
rQuality.vMSE = nanmean(mDiffMat2);

% RMSE berechnen
rQuality.vRMSE = rQuality.vMSE.^(1/2);

% Mean Absolute Error
rQuality.vMAE = nanmean(mAbsDiffMat);

% Mean Absolute Percentage Error
rQuality.vMAPE = nanmean( abs(mDiffMat ./ mYtrue) );

% Korrelation berechen und Regressionen durchfuehren 
rQuality.vCor = NaN(1,iNumVars);
vAlpha = NaN(1,iNumVars);
vBeta = NaN(1,iNumVars);
vR2 = NaN(1,iNumVars);
vF = NaN(1,iNumVars);
vP = NaN(1,iNumVars);
vErrVar = NaN(1,iNumVars);
vStdYtrue = nanstd(mYtrue);
vStdYest = nanstd(mYest);
for iIdxI=1:iNumVars
    % Korrelation berechnen
    mTemp = [mYtrue(:,iIdxI),mYest(:,iIdxI)];
    mCor = nancov(mTemp,'pairwise');
    rQuality.vCor(iIdxI) = mCor(1,2) / (vStdYtrue(iIdxI) * vStdYest(iIdxI));
    % Regression durchfuehren
    lIsNaN = isnan(mYtrue(:,iIdxI)) | isnan(mYest(:,iIdxI));
    vYtemp = mYtrue(~lIsNaN,iIdxI);
    % Sicherheitspruefung, ob Prognose konstant ist
    if std(mYest(~lIsNaN,iIdxI)) < 1e-6
        % Wenn ja, keine Regression durchfuehren
        vAlpha(iIdxI) = NaN;
        vBeta(iIdxI) = NaN;
        vR2(iIdxI) = NaN;
        vF(iIdxI) = NaN;
        vP(iIdxI) = NaN;
        vErrVar(iIdxI) = NaN;
    else
        % Konstante hinzufuegen
        mXtemp = [ones(sum(~lIsNaN),1), mYest(~lIsNaN,iIdxI)];
        [b,~,~,~,stats] = regress(vYtemp,mXtemp);
        vAlpha(iIdxI) = b(1);
        vBeta(iIdxI) = b(2);
        vR2(iIdxI) = stats(1);
        vF(iIdxI) = stats(2);
        vP(iIdxI) = stats(3);
        vErrVar(iIdxI) = stats(4);
    end
end
rQuality.rRegResults.vAlpha = vAlpha;
rQuality.rRegResults.vBeta = vBeta;
rQuality.rRegResults.vR2 = vR2;
rQuality.rRegResults.vF = vF;
rQuality.rRegResults.vP = vP;
rQuality.rRegResults.vErrVar = vErrVar;

% Trefferquote berechnen
mHits = sign(mYtrue) == sign(mYest);
rQuality.vHitRate = nanmean(mHits);

% Wegstrecke berechnen
mProfits = sign(mYest) .* mYtrue;
vCumProfits = nansum(mProfits);
vMaxProfits = nansum(abs(mYtrue));
rQuality.vProfitRate = vCumProfits ./ vMaxProfits;

% Daten fuer Tabelle zusammenstellen
if nargout > 1
    ForecastVar = [rQuality.vMSE; rQuality.vRMSE; rQuality.vMAE; ...
        rQuality.vCor; rQuality.vHitRate; rQuality.vProfitRate; ...
        rQuality.rRegResults.vAlpha; rQuality.rRegResults.vBeta; ...
        rQuality.rRegResults.vR2; rQuality.rRegResults.vF; ...
        rQuality.rRegResults.vP; rQuality.rRegResults.vErrVar];
    cRowNames = {'Mean Squared Error';'Rooted Mean Squared Error';'Mean Absolute Error'; ...
        'Correlation True,Est'; 'Hit rate'; 'Profit rate'; ...
        'Regression Alpha'; 'Regression Beta'; ...
        'Regression R2'; 'Regression F-stat'; ...
        'Regression p-value'; 'Regression Error Variance'};
    tTable = array2table(ForecastVar,'RowNames',cRowNames);
end
end

