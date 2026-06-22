function val = fCfg(sName, default)
%FCFG Read an INF_<NAME> environment override, else return the literal default.
%   val = FCFG('COUNTRY', 'US') returns getenv('INF_COUNTRY') if set, else 'US'.
%   The value is cast to the class of `default` (logical/numeric/char), so a
%   direct run with no env var behaves exactly as the CONFIG literal specifies.
%
%   This lives in the project root (always on the MATLAB path) so it is callable
%   from a script's CONFIG block before any addpath. It reads the process
%   environment, which survives the `clear` at the top of each script, allowing
%   batch sweeps to be driven by INF_* variables across separate MATLAB calls.
    sRaw = getenv(['INF_', sName]);
    if isempty(sRaw)
        val = default;
        return
    end
    if islogical(default)
        val = any(strcmpi(sRaw, {'1', 'true', 'yes', 'y'}));
    elseif isnumeric(default)
        val = str2double(sRaw);
    else
        val = sRaw;
    end
end
