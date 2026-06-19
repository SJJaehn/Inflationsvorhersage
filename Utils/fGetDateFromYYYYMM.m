function dtDates = fGetDateFromYYYYMM(yyyymm, sMode)
% Function to transform dates from yyyymm to dates
%
% Input:
%       yyyymm:     T x 1 vector of numeric dates
%       sMode:      String, specifies whether to transform to the 'start'
%                   or 'end' of a month (default = 'end')
%
% Output:
%       dtDates:    T x 1 datetime array

% Check input arguments
arguments
    yyyymm (:,1) {mustBeNumeric, mustBeNonnegative}
    sMode string = 'end'
end

% 1. Step: Transform numeric to strings
sYYYYMM = num2str(yyyymm);

% 2. Step: Assign each date to a cell
cYYYYMM = cellstr(sYYYYMM);

% 3. Step: Convert to date
dtDates = datetime(cYYYYMM, 'InputFormat','yyyyMM');

% 4. Step: Convert to end of month if required
dtDates = dateshift(dtDates, sMode, 'month');
end