function [MODEL] = rvr_train(x, t, OPTIONS)
% RVM_TRAIN Trains an RVM Model using SB1 Tipping Toolbox
%
%   input ----------------------------------------------------------------
%
%       o X        : (N x D), N  input data points of D dimensionality.
%
%       o t        : (N x 1), N  output data points
%
%       o options     : struct
%
%
%   output ----------------------------------------------------------------
%
%       o model       : struct.
%
%
%% %    RVM OPTIONS
%       KERNEL  Kernel type: see SB1_KERNELFUNCTION for options
%       LEN     Kernel length scale
%       USEBIAS Set to non-zero to utilise a "bias" offset
%       MAXITS  Maximum iterations to run for.

%% Parse RVM Options
% Transform Data to Columns
x = x(:);
t = t(:);

% Parsing Parameter for RVR
useBias             = OPTIONS.useBias;
kernel              = OPTIONS.kernel;
lengthScale         = OPTIONS.lengthScale; 
maxIts              = OPTIONS.maxIts;
%monIts              = round(maxIts/20);

%% Set environment settings
USER_OPTIONS  = SB2_UserOptions('iterations', maxIts, 'diagnosticLevel', 0, 'monitor', 10);
SETTINGS = SB2_ParameterSettings('NoiseStd', 0.1);

%% Compute basis matrix
if isempty(OPTIONS.BASIS),
    OPTIONS.BASIS = SB2_KernelFunction(x, x, kernel, lengthScale);
    M = size(OPTIONS.BASIS,2);
    % Add bias vector if necessary
    if useBias
       OPTIONS.BASIS = [OPTIONS.BASIS ones(M,1)]; 
    end
end
%% Train RVR Model
% "Train" a sparse Bayes kernel-based model (relevance vector machine) 
[PARAMETER, HYPERPARAMETER, DIAGNOSTIC] = ...
    SparseBayes('Gaussian', OPTIONS.BASIS, t, USER_OPTIONS, SETTINGS);

%% Model Parameters
%       WEIGHTS Parameter values of estimated model (sparse)
%       KERNEL  Type of kernel used in the model
%       
%       USED    Index vector of "relevant" kernels (data points)
%       BIAS    Value of bias or offset parameter
%       ML      Log marginal likelihood of model
%       ALPHA   Estimated hyperparameter values (sparse)
%       BETA    Estimated inverse noise variance for regression
%       GAMMA   "Well-determinedness" factors for relevant kernels

MODEL.weights       = PARAMETER.Value;
MODEL.kernel        = OPTIONS.kernel;
MODEL.lengthScale   = OPTIONS.lengthScale;
MODEL.RVs_idx       = PARAMETER.Relevant;
MODEL.useBias       = OPTIONS.useBias;
MODEL.marginal      = DIAGNOSTIC.Likelihood;
MODEL.alpha         = HYPERPARAMETER.Alpha;
MODEL.beta          = HYPERPARAMETER.beta;
MODEL.gamma         = DIAGNOSTIC.Gamma;
MODEL.RVs           = x(MODEL.RVs_idx,:);
MODEL.SIGMA         = DIAGNOSTIC.SIGMA;

end