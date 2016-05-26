function [y_rvm] = svr_predict(x,  MODEL)
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
weights     = MODEL.sv_coef;
kernel      = MODEL.kernel;
lengthScale = MODEL.lengthScale;
SVs_idx     = MODEL.sv_indices;
SVs         = MODEL.SVs;
bias        = -MODEL.rho;

% Compute RVM over test data and calculate error
BASIS	= KernelFunction(x, SVs, kernel, lengthScale);

% Add bias vector if necessary
y_rvm	= BASIS*weights + bias;

end