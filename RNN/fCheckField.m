function rStruct = fCheckField(rStruct, sField, default)
% Function for checking the existence of a field and for assigning default
% values if non existent

if ~isfield(rStruct,sField)
    % Set default if missing
    rStruct.(sField) = default;   
else
    rStruct.(sField) = rStruct.(sField);
end
end