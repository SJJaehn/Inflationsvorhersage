% Script for plotting the benchmark

% Clear console
clear; clc; close all;

% Set path
sOldPath = path;
addpath('./Utils/');

% Load data
load('./DATA/Empirical/Market.mat');

% Define market return
vMktRet = CRSPSP_M;

% Date is stored as yyyymm, which refers to the end-of-month. Convert to
% actual date
dtDates = fGetDateFromYYYYMM(yyyymm);

% Calculate cumulative return
vCumRet = cumsum(vMktRet);

% Plot cumulative returns (percentage)
plot(dtDates, vCumRet * 100,'LineWidth',1.5,'Color','black');
box off;
ylabel('Cumulative Return (in \%)','Interpreter','latex','FontSize',12);

% Alternatively: Use custom function
fPlotCumRet(dtDates, vMktRet, [], 'Name','S&P500');

% Restore path
path(sOldPath);