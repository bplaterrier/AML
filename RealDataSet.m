close all;
clear;
clc;

%% Load the real data
load housing.mat

t = MEDV;
x = [CRIM ZN INDUS CHAS NOX RM AGE DIS RAD TAX PTRATIO B LSTAT];
x = normalize(x);

%% SVR
OPTIONS.svr_type        = 0;            % 0: epsilon-SVR / 1: nu-SVR
OPTIONS.kernel_type     = 2;            % 1: linear / 2: gaussian / 3: polyN / 4: precomputed kernel matrix
OPTIONS.kernel          = 'gaussian';   % kernel type, used for custom basis matrix
OPTIONS.C               = 50;           % penalty factor (default 1)
OPTIONS.nu              = 0.05;         % nu parameter (default 1)
OPTIONS.epsilon         = 10;          % epsilon parameter (default 0.1)
OPTIONS.tolerance       = 0.001;        % tolerance of termination criterion (default 0.001)
OPTIONS.lengthScale     = 10;          % lengthscale parameter (~std dev for gaussian kernel)
OPTIONS.probabilities   = 0;            % whether to train a SVR model for probability estimates, 0 or 1 (default 0);
OPTIONS.useBias         = 0;            % add bias to the model (for custom basis matrix)

MODEL = svr_train(t, x, OPTIONS);
[y_svm] = svr_predict(x, MODEL);


[sortedX, index] = sort(x(:,1));
plot(sortedX, y_svm(index))

gfit2(t, y_svm, '2')


%% RVR

% Choose kernel function
OPTIONS.kernel = 'gaussian';        % choose kernel function (see SB2_KernelFunction.m)
OPTIONS.lengthScale = 10;          % ~std. dev. for gaussian kernel
OPTIONS.useBias = 0;                % add bias vector to the estimated model
OPTIONS.maxIts = 500;
OPTIONS.BASIS = [];

% Train the model
[MODEL] = rvr_train(x, t, OPTIONS);

% Predict values
[y_rvm, mu_star, sigma_star] = rvr_predict(x, MODEL);


[sortedX, index] = sort(x(:,1));
plot(sortedX, y_rvm(index))