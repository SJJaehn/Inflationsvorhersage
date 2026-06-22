function sDir = fResultDir(sRoot, sType, sCountry, sMode, sOptions)
%FRESULTDIR Build (and create) the structured results directory and return it.
%   sDir = FRESULTDIR(sRoot, sType, sCountry, sMode, sOptions) returns
%   <sRoot>/<sType>/<sCountry>/<sMode>/<sOptions>/ , creating it if needed.
%
%   The layout mirrors the Python port (see python/util.py result_dir):
%       sType    : model family ('single','full','AR','VAR','PCA','PLS', ...)
%       sCountry : 'US' or 'UK' (use fCountryFromPath on the input CSV path)
%       sMode    : 'insample' or 'oos'
%       sOptions : a chain of the run options, e.g. 'train60_rolling_lag1'
%
%   Files written there (results.csv, chart.png, predictions.csv, ...) use
%   fixed names, so a new run for the same configuration overwrites the
%   previous one.
    sDir = fullfile(sRoot, sType, sCountry, sMode, sOptions);
    if ~exist(sDir, 'dir')
        mkdir(sDir);
    end
end
