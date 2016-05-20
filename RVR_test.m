close all;
clear;
clc;

%% Generate the data
% Fix the random seed for reproductibility of results
rseed = 1; % choose a seed

% Set default values for data
data.N = 200;                   % number of samples
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

% Choose kernel function
OPTIONS.kernel = 'gaussian';        % choose kernel function (see SB2_KernelFunction.m)
OPTIONS.lengthScale = 0.22;          % ~std. dev. for gaussian kernel
OPTIONS.useBias = 0;                % add bias vector to the estimated model
OPTIONS.maxIts = 500;

% Normalize the data
x_rvm = normalize(x);

% Train the model
[MODEL] = rvr_train(x_rvm, t, OPTIONS);

% Predict values
y_rvm = rvr_predict(x_rvm, MODEL);

figure
if data.D==1,
    plot(x, y,'-', 'Color', COL_sinc);
    hold on
    plot(x, t, '.', 'Color', COL_data);
    plot(x, y_rvm,'r-','LineWidth', 1, 'Color', COL_pred);
  
    plot(x(MODEL.RVs_idx), t(MODEL.RVs_idx),'o', 'Color', COL_rv);
else
    mesh(gx, gy, reshape(y_rvm,size(gx)), 'edgecolor', COL_sinc, 'facecolor', COL_sinc);
    mesh(gx, gy, reshape(t,size(gx)), 'edgecolor', COL_sinc, 'facecolor' ,COL_data);
end
hold off
legend('Actual Model', 'Datapoints', 'Regression', 'Relevance Vectors', 'Location', 'NorthWest')

%% Do RVR on the data (rvm_regressor)

% Choose kernel function
OPTIONS.kernel = 'gaussian';        % choose kernel function (see SB2_KernelFunction.m)
OPTIONS.lengthScale = 0.22;          % ~std. dev. for gaussian kernel
OPTIONS.useBias = 0;                % add bias vector to the estimated model
OPTIONS.maxIts = 500;

% Normalize the data
x_rvm = normalize(x);

% Train and predict
MODEL = [];
[y_rvm, MODEL] = rvm_regressor(x_rvm, t, OPTIONS, MODEL);

plotRVR(x,y,y_rvm,t,data,MODEL,COLOR);

