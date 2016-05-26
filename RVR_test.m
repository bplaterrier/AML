close all;
clear;
clc;

%% Generate the data
% Fix the random seed for reproductibility of results
rseed = 3; % choose a seed

% Set default values for data
data.N = 600;                   % number of samples
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

%% Do RVR on the data
clc
% Choose kernel function
OPTIONS.kernel = 'gaussian';        % choose kernel function (see SB2_KernelFunction.m)
OPTIONS.lengthScale = 0.01;          % ~std. dev. for gaussian kernel
OPTIONS.useBias = 0;                % add bias vector to the estimated model
OPTIONS.maxIts = 500;
OPTIONS.BASIS = [];

% Normalize the data
x_rvm = x;
x_rvm = normalize(x);

% Train the model
[MODEL] = rvr_train(x_rvm, t, OPTIONS);

% Predict values
[y_rvm, mu_star, sigma_star] = rvr_predict(x_rvm, MODEL);
MODEL.mu_star = mu_star;
MODEL.sigma_star = sigma_star;

plotRVR(x,y,y_rvm,t,data,MODEL,COLOR);
axis([-inf inf -1 1.5])
%% Do RVR on the data (rvm_regressor)
clc
% Choose kernel function
OPTIONS.kernel = 'gaussian';        % choose kernel function (see SB2_KernelFunction.m)
OPTIONS.lengthScale = 0.01;          % ~std. dev. for gaussian kernel
OPTIONS.useBias = 0;                % add bias vector to the estimated model
OPTIONS.maxIts = 500;

% Normalize the data
x_rvm = normalize(x);

% Train and predict
MODEL = [];
[y_rvm, MODEL] = rvm_regressor(x_rvm, t, OPTIONS, MODEL);

plotRVR(x,y,y_rvm,t,data,MODEL,COLOR);
axis([-inf inf -1 1.5])
% Compute errors
mse  = gfit2(t, y_rvm, '1');
nmse = gfit2(t, y_rvm, '2');
disp([mse nmse]);
