function rModel = fParaVecToMat(rModel, vPara)
% Function for reshaping the vector of free parameters of a recurrent
% neural network


% Extract OMEGA
iNumOmega = sum(rModel.rPrms.lOmega,'all');
rModel.rPrms.mOmega(rModel.rPrms.lOmega) = vPara(1:iNumOmega)';
    

if rModel.rErrCorrLayer.iNumUnits > 0
    % Extract THETA
    iNumTheta = sum(rModel.rPrms.lTheta,'all');
    rModel.rPrms.mTheta(rModel.rPrms.lTheta) = vPara(iNumOmega+1:iNumOmega+iNumTheta);
    iNumOmega = iNumOmega + iNumTheta;

    % Extract BIAS
    iNumBias = sum(rModel.rPrms.lBiasEC,'all');
    rModel.rPrms.vBiasEC(rModel.rPrms.lBiasEC) = vPara(iNumOmega+1:iNumOmega+iNumBias);
    iNumOmega = iNumOmega + iNumBias;

    % Extract OMEGA Error Correction
    iNumOmegaEC = sum(rModel.rPrms.lOmegaEC,'all');
    rModel.rPrms.mOmegaEC(rModel.rPrms.lOmegaEC) = vPara(iNumOmega+1:iNumOmega+iNumOmegaEC);
    iNumOmega = iNumOmega + iNumOmegaEC;
end
    
% Extract AR
iNumAR = sum(rModel.rPrms.lAR,'all');
rModel.rPrms.mAR(rModel.rPrms.lAR) = vPara(iNumOmega+1:iNumOmega+iNumAR);

% Extract INIT
iNumUnitInit = sum(rModel.rPrms.lUnitInit,'all');
rModel.rPrms.vUnitInit(rModel.rPrms.lUnitInit) = ...
    vPara(iNumOmega+iNumAR+1:iNumOmega+iNumAR+iNumUnitInit);

% Extract BETA
iNumBeta = sum(rModel.rPrms.lBeta,'all');
rModel.rPrms.mBeta(rModel.rPrms.lBeta) = ...
    vPara(iNumOmega+iNumAR+iNumUnitInit+1:iNumOmega+iNumAR+iNumUnitInit+iNumBeta);
end