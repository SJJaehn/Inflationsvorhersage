function cCell = fMatrix2Cell(mMatrix, iNumDigits, lReplaceNaN)
% Function for converting a matrix into a cell-array
%
% Input:
%   mMatrix:        R x C data matrix
%                   R: number of rows
%                   C: number of columns
%   iNumDigits:     Scalar, integer, number of digits (default = 2)
%   lReplaceNaN:    Logical, indicates whether to remove missing values
%                   (default = true)
%
% Output:
%   cCell:          R x C data matrix, converted to cell-array
%                   R: number of rows
%                   C: number of columns

% Check input arguments
arguments
    mMatrix {mustBeNumeric}
    iNumDigits (1,1) {mustBeNumeric, mustBeNonnegative} = 2
    lReplaceNaN (1,1) {mustBeNumericOrLogical, mustBeNonnegative}= true
end

% Round data
mMatrix = round(mMatrix, iNumDigits);

% Find missing values
lIsNaN = isnan(mMatrix);

% To cell
cCell = sprintfc(['%.',num2str(iNumDigits),'f'], mMatrix);

% Replace missing values
if lReplaceNaN
    cCell(lIsNaN) = {''};
end
end