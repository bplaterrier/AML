function [y_svm, MODEL] = svm_regressor(x, t, OPTIONS, MODEL)
%SVM_REGRESSOR.
%
%   input ----------------------------------------------------------------
%
%       o X            : (N x D), number of input datapoints.
%
%       o t            : (N x 1), number of output datapoints.
%
%       o svr_options  : (1 x 1), parameter settings for svr model
%
%   output----------------------------------------------------------------
%
%       o model        : struct, result of linear input-output function
%
%           
%
%
%%

if isempty(t)
    t = randn(length(x),1);
end

% if the model doesn't exist, train it
if isempty(MODEL)
        % Train SVM Regressor
        MODEL = svr_train(t, x, OPTIONS);
end

% Predict Values based on query points
[y_svm]  = svr_predict(x, MODEL);

end