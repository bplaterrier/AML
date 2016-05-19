function [y_rvm] = rvr_predict(x,  MODEL)
% RVR_PREDICT % Predicts labels and computes accuracy for new
% data with a learnt RVM model
%   input ----------------------------------------------------------------
%
%       o X        : (N x D), N  input data points of D dimensionality.
%       o MODEL    : struct
%
%
%
%   output ----------------------------------------------------------------
%
%       o y_rvm            : (N x 1), decision values
%
%
%% Predict labels given model and data

% Test
weights     = MODEL.weights;
kernel      = MODEL.kernel;
lengthScale = MODEL.lengthScale;
useBias     = MODEL.useBias;
RVs_idx     = MODEL.RVs_idx;
RVs         = MODEL.RVs;

% Compute RVM over test data and calculate error
BASIS	= SB2_KernelFunction(x(:), RVs, kernel, lengthScale);
y_rvm	= BASIS*weights;

end