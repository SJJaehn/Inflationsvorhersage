function fPlotCumRet(dtDates,vX,vB,options)
% Function for plotting the cumulative returns of two strategies
%
% Input:
%   dtDates:        T x 1 datetime array
%                   T: number of time-series observations
%   vX:             T x 1 vector of returns
%                   T: number of time-series observations
%   vB:             T x 1 vector of benchmark returns
%                   T: number of time-series observations
%   options:        Name-Value pair arguments
%       'UseLog':       Logical, specifies whether to use log returns
%                       (default = false)
%       'Name':         String, specifies the name of the strategy

% Check input arguments
arguments
    dtDates (:,1)
    vX (:,1) {mustBeNumeric}
    vB (:,1) {mustBeNumeric} = []
    options.UseLog (1,1) {mustBeNumericOrLogical} = false
    options.Name char = 'Strategy'
end

% Merge returns
mX = [vX, vB];

% Remove missing values
lIsNaN          = any(isnan(mX),2);
mX(lIsNaN,:)    = [];
dtDates(lIsNaN) = [];

% Calculate cumulative returns
if options.UseLog
    % Log returns
    mCumRet = cumprod(1 + mX) - 1;         
else
    % Discrete returns
    mCumRet = cumsum(mX);
end

% Make to percentage
mCumRet = mCumRet * 100;

% Plot
if size(mX,2) == 2
    % Benchmark is available

    % Plot returns
    plot(dtDates, mCumRet(:,1),'LineWidth',1.5,'Color','black');
    hold on
    plot(dtDates, mCumRet(:,2),'LineWidth',1.5,'Color','black');
    hold off

    % Legend
    legend({options.Name,'Benchmark'},'Box','off','Location','best','FontSize',12)
else
    % No benchmark available

    % Plot returns
    plot(dtDates, mCumRet(:,1),'LineWidth',1.5,'Color','black');

    % Legend
    legend({options.Name},'Box','off','Location','best','FontSize',12)
end
box off;
ylabel('Cumulative Return (in \%)','Interpreter','latex','FontSize',12);
end



