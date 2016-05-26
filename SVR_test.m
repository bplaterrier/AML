close all;
clear;
clc;

%% Generate the data

% Fix the random seed for reproductibility of results
rseed = 1; % choose a seed

% Set default values for data
data.N = 100;                   % number of samples
data.D = 1;                     % dimension of data
data.scale = 10;                % scale of dimensions
data.noise = 0.1;               % scale of noise
data.noiseType = 'gauss';       % type of noise ('gauss' or 'unif')

% Generate the sinc data
[x, y, t] = generateSinc(data, rseed); 

% Plot settings
COLOR.sinc = 'k';     % color of the actual function
COLOR.data = 'b';     % color of the real data
COLOR.pred = 'r';     % color of the prediction
COLOR.rv = 'k';       % color of the relevance vectors

%% Plot the data and actual function
plotModel(x, y, t, data, COLOR);

%% Do SVR on the data

% Perform a scaling of the data (normalisation)
x_svm = normalize(x);

% Train the model on the data
OPTIONS.svr_type        = 0;            % 0: epsilon-SVR / 1: nu-SVR
OPTIONS.kernel_type     = 2;            % 1: linear / 2: gaussian / 3: polyN / 4: precomputed kernel matrix
OPTIONS.kernel          = 'gaussian';   % kernel type, used for custom basis matrix
OPTIONS.C               = 5;           % penalty factor (default 1)
OPTIONS.nu              = 0.05;         % nu parameter (default 1)
OPTIONS.epsilon         = 0.2;          % epsilon parameter (default 0.1)
OPTIONS.tolerance       = 0.001;        % tolerance of termination criterion (default 0.001)
OPTIONS.lengthScale     = 0.2;          % lengthscale parameter (~std dev for gaussian kernel)
OPTIONS.probabilities   = 0;            % whether to train a SVR model for probability estimates, 0 or 1 (default 0);
OPTIONS.useBias         = 0;            % add bias to the model (for custom basis matrix)

% Train the model
MODEL = svr_train(t, x_svm, OPTIONS);

[y_svm, MODEL] = svm_regressor(x_svm, t, OPTIONS, MODEL);

% Do SVR on data using svmregressor
% [y_svm] = svr_predict(x_svm, MODEL);

%Plot the results
plotSVR(x, y, y_svm, t, data, MODEL, OPTIONS, COLOR)

% Compute errors
mse  = gfit2(t, y_svm, '1');
nmse = gfit2(t, y_svm, '2');
disp([mse nmse]);

