function rModel = fInitPrms(rModel)
% Function for initializing the parameters

% Get dimensions
iNumUnits = rModel.iNumUnits;
iNumBetas = iNumUnits + rModel.rOutLayer.lEstAlpha;
iNumResp = rModel.iNumResp;
iNumVars = rModel.iNumVars + rModel.rHidLayer.lEstBias;

% Initialize parameters that map input to hidden layers
rModel.rPrms.mOmega = 0.01 * randn(iNumVars, iNumUnits);
rModel.rPrms.lOmega = true(iNumVars, iNumUnits);

% Initialize parameters that map previous inputs to hidden layers
rModel.rPrms.mAR = 0.01 * randn(iNumUnits, iNumUnits);
rModel.rPrms.lAR = true(iNumUnits, iNumUnits);

% Initialize hidden units
rModel.rPrms.vUnitInit = 0.01 * randn(iNumUnits, 1);
rModel.rPrms.lUnitInit = true(iNumUnits,1);

% Initialize betas (weights that map hidden units to output)
rModel.rPrms.mBeta = 0.01 * randn(iNumResp, iNumBetas);
rModel.rPrms.lBeta = true(iNumResp, iNumBetas);

% Initialize error correction layer
if rModel.rErrCorrLayer.iNumUnits > 0
    % Initialize parameters that map previous model error to error
    % correction network
    rModel.rPrms.mTheta = 0.01 * randn(iNumResp, rModel.rErrCorrLayer.iNumUnits);
    rModel.rPrms.lTheta = true(iNumResp, rModel.rErrCorrLayer.iNumUnits);
    
    % Bias of error correction network
    rModel.rPrms.vBiasEC = 0.01 * randn(rModel.rErrCorrLayer.iNumUnits, 1);
    rModel.rPrms.lBiasEC = true(rModel.rErrCorrLayer.iNumUnits, 1);
        
    % Parameters to map output of error correction network to hidden units
    rModel.rPrms.mOmegaEC = 0.01 * randn(rModel.rErrCorrLayer.iNumUnits, iNumUnits);
    rModel.rPrms.lOmegaEC = true(rModel.rErrCorrLayer.iNumUnits, iNumUnits);
    
else
    rModel.rPrms.mOmegaEC = [];
    rModel.rPrms.lOmegaEC = [];
    rModel.rPrms.mTheta = [];
    rModel.rPrms.lTheta = [];
    rModel.rPrms.vBiasEC = [];
    rModel.rPrms.lBiasEC = [];
end

% Initialize memory for gradients
cFieldnames = fieldnames(rModel.rPrms);
lDropField = cellfun(@(x)strcmp('l',x(1)),cFieldnames,'UniformOutput',true);
cFieldnames(lDropField) = [];
for iIdxF = 1:length(cFieldnames)
    rModel.rGrad.(cFieldnames{iIdxF}) = zeros(size(rModel.rPrms.(cFieldnames{iIdxF})));
end
end