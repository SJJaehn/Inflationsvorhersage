% Script for demonstraing the usage of PCA. This script generates true
% (principal components) and noisy variables and extracts the PCs from the
% set of noisy variables.

% Clear console
clear; clc; close all;

% Set path
sOldPath = path;
addpath('./Utils');

% Random number generator
rng(42);

% Settings
iNumObs         = 1000;     % Number of observations
iNumVarsTrue    = 4;        % Number of true variables
iNumVarsNoise   = 10;       % Number of noisy variables
dNoise          = 0.2;      % Noise factor

% Generate true components (with decreasing variance)
mPCs = randn(iNumObs, iNumVarsTrue) .* linspace(1,0.5,iNumVarsTrue);

% Generate noisy variables
mLoadings = randn(iNumVarsNoise, iNumVarsTrue);
mX        = mPCs * mLoadings' + dNoise * randn(iNumObs, iNumVarsNoise);

% Estimate full PCA with all components
rModel = fEstPCA(mX, "iNumComp", iNumVarsNoise);

% The function returns loadings, scores, and eigenvectors as well as the
% cumulative variance explained. Plot the explained variance
h1 = figure(1);
plot(1:iNumVarsNoise, rModel.vExplVar, 'LineWidth',2,'Color','black');
ylabel('Explained Variance');
xlabel('Number of Components');

% Restore path
path(sOldPath);