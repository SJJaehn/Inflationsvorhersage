% Ensure joint stability via companion matrix scaling
function mPhi = fMakeStableVAR(mPhi)
    [k, ~, p] = size(mPhi);
    stable = false;
    while ~stable
        % Build companion matrix
        M = zeros(k*p);
        M(1:k, :) = reshape(mPhi, k, []);
        M(k+1:end, 1:end-k) = eye(k*(p-1));

        % Check spectral radius
        maxEig = max(abs(eig(M)));
        if maxEig < 0.99
            stable = true;
        else
            mPhi = mPhi * 0.95;  % Scale down all lags uniformly
        end
    end
end