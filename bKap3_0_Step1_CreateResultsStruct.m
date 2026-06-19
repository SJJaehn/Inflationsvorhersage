% This is a modified script from GWZ to extract the data in a more
% user-friendly format

clear;clc;

load P00Data;

temp=struct('Name',[],'SampleBeg',[], 'SampleEnd',[], 'OOSBeg',[], 'OOSEnd',[]);
Results=repmat(temp,46,1);

% Get number of predictors
iNumVars = size(VARS,1);

% Convert struct to cell
cVARS           = struct2cell(VARS);
cFieldnames     = fieldnames(VARS);

% Get all predictor names
cVarNames = cVARS(strcmpi(cFieldnames,'Name'),:);

% Count number of predictors
iIdxFreq = find(strcmpi(cFieldnames,'Freq'));
iNumVarsM   = sum(strcmpi(cVARS(iIdxFreq,:),'Monthly'));
iNumVarsQ   = sum(strcmpi(cVARS(iIdxFreq,:),'Quarterly'));
iNumVarsSA  = sum(strcmpi(cVARS(iIdxFreq,:),'Semiannual'));
iNumVarsA   = sum(strcmpi(cVARS(iIdxFreq,:),'Annual'));
assert(iNumVars == (iNumVarsA+iNumVarsSA+iNumVarsQ+iNumVarsM),'Number of predictors must agree');

% Determine number of observations
iIdxData = find(strcmpi(cFieldnames,'DataIS'));
iNumObsM = size(cVARS{iIdxData, find(strcmpi(cVARS(iIdxFreq,:),'Monthly'),1,'first')},1);
iNumObsQ = size(cVARS{iIdxData, find(strcmpi(cVARS(iIdxFreq,:),'Quarterly'),1,'first')},1);
iNumObsSA = size(cVARS{iIdxData, find(strcmpi(cVARS(iIdxFreq,:),'Semiannual'),1,'first')},1);
iNumObsA = size(cVARS{iIdxData, find(strcmpi(cVARS(iIdxFreq,:),'Annual'),1,'first')},1);

% Initialize memory for all predictors
mXmonthly       = NaN(iNumObsM, iNumVarsM);
mXquarterly     = NaN(iNumObsQ, iNumVarsQ);
mXsemiannually  = NaN(iNumObsSA, iNumVarsSA);
mXannually      = NaN(iNumObsA, iNumVarsA);
cXnamesM        = cell(iNumVarsM,1);
cXnamesQ        = cell(iNumVarsQ,1);
cXnamesSA       = cell(iNumVarsSA,1);
cXnamesA        = cell(iNumVarsA,1);

% Initialize counter
iCounterM       = 1;
iCounterQ       = 1;
iCounterSA      = 1;
iCounterA       = 1;

% Extract predictors
for i = 1:iNumVars
    Results(i).SampleBeg = VARS(i).SampleBeg;
    Results(i).SampleEnd = VARS(i).SampleEnd;
    Results(i).Name = VARS(i).Name;
    if strcmp(VARS(i).Freq,'Monthly')
        Results(i).OOSBeg = Results(i).SampleBeg + 2000;

        % Save predictor
        if strcmpi('rsvix', cVarNames{i})
            mXmonthly(:,iCounterM) = VARS(i).DataIS{1};
        elseif any(strcmpi({'sntm','fbm'}, cVarNames{i}))
            mXmonthly(:,iCounterM) = VARS(i).DataIS{5};  % Warning! This is discrete return. 5 would be log return
        else
            mXmonthly(:,iCounterM) = VARS(i).DataIS;
        end
        cXnamesM{iCounterM} = cVarNames{i};
        iCounterM = iCounterM + 1;

    elseif strcmp(VARS(i).Freq,'Quarterly')
        Results(i).OOSBeg = Results(i).SampleBeg + 200;

        % Save predictor
        mXquarterly(:,iCounterQ) = VARS(i).DataIS;
        cXnamesQ{iCounterQ} = cVarNames{i};
        iCounterQ = iCounterQ + 1;
    elseif strcmp(VARS(i).Freq,'Semiannual')
        Results(i).OOSBeg = Results(i).SampleBeg + 200;

        % Save predictor
        mXsemiannually(:,iCounterSA) = VARS(i).DataIS;
        cXnamesSA{iCounterSA} = cVarNames{i};
        iCounterSA = iCounterSA + 1;
    elseif strcmp(VARS(i).Freq,'Annual')
        Results(i).OOSBeg = Results(i).SampleBeg + 20;

        % Save predictor
        mXannually(:,iCounterA) = VARS(i).DataIS;
        cXnamesA{iCounterA} = cVarNames{i};
        iCounterA = iCounterA + 1;
    end
    Results(i).OOSEnd = Results(i).SampleEnd;
end

% Save data
save('PredictorDataGWZ.mat','-regexp','cXnames','mX')

% Change some OOSBeg dates manually for some variables that start in the 1990s
Results(2).OOSBeg = Results(2).SampleBeg + 1000; % vp
Results(3).OOSBeg = Results(3).SampleBeg + 1000; %impvar
Results(4).OOSBeg = Results(4).SampleBeg + 1000; % vrp
Results(22).OOSBeg = Results(22).SampleBeg + 1000; % svix
Results(8).OOSBeg = Results(8).SampleBeg + 100;  % crdstd

save('ResultsFile.mat','Results', 'cVarNames');