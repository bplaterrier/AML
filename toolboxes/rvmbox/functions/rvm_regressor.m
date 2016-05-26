function [y_rvm, MODEL] = rvm_regressor(x,t, OPTIONS, MODEL)
%RVM_REGRESSOR.
%
%   input ----------------------------------------------------------------
%
%       o X            : (N x D), number of input datapoints.
%
%       o y            : (N x 1), number of output datapoints.
%
%       o OPTIONS      : (1 x 1), parameter settings for svr model
%
%   output----------------------------------------------------------------
%
%       o MODEL        : struct, result of linear input-output function
%
%           
%
%
%%

if isempty(t)
    t = randn(length(x),1);
end

% Transform data to columns
% x = x(:);
% t = t(:);

% if the model doesn't exist, train it
if isempty(MODEL)
        % Train SVM Regressor
    [MODEL] = rvr_train(x, t, OPTIONS);
end

% Predict RVR function from Train data
[y_rvm, mu_star, sigma_star] = rvr_predict(x,  MODEL);
MODEL.mu_star = mu_star;
MODEL.sigma_star = sigma_star;
end