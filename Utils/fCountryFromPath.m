function sCountry = fCountryFromPath(sPath)
%FCOUNTRYFROMPATH Pull the country code ('US'/'UK') out of a data path.
%   sCountry = FCOUNTRYFROMPATH('./DATA/Liedtke/US/aggregated.csv') -> 'US'.
%   Returns 'NA' if no US/UK path component is found.
    sCountry = 'NA';
    cParts = strsplit(sPath, {'/', '\'});
    for i = 1:numel(cParts)
        sU = upper(cParts{i});
        if strcmp(sU, 'US') || strcmp(sU, 'UK')
            sCountry = sU;
            return
        end
    end
end
