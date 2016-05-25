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
data.noise = 0.2;               % scale of noise
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

%% Do RVR on the data (rvm_regressor)
close all
% Choose kernel function
OPTIONS.kernel = 'gaussian';        % choose kernel function (see SB2_KernelFunction.m)
OPTIONS.lengthScale = 2.2;          % ~std. dev. for gaussian kernel
OPTIONS.useBias = 0;                % add bias vector to the estimated model
OPTIONS.maxIts = 500;

% Normalize the data
x_rvm = x;%normalize(x);

% Train and predict
MODEL = [];
[y_rvm, MODEL] = rvm_regressor(x_rvm, t, OPTIONS, MODEL);

plotRVR(x,y,y_rvm,t,data,MODEL,COLOR);
axis([-inf inf -1 1.5])
% Compute errors
mse  = gfit2(t, y_rvm, '1');
nmse = gfit2(t, y_rvm, '2');
% disp([mse nmse]);

%% Plot the regression model away from the original data
new_data = data;
new_data.N = 300;                   % number of samples
new_data.scale = 20;                % scale of dimensions

[newX, newY, ~] = generateSinc(new_data, rseed);

% newX_rvm = normalize(newX);

[newY_rvm, mu_star, sigma_star] = rvr_predict(newX, MODEL);
MODEL.mu_star = mu_star;
MODEL.sigma_star = sigma_star;

% Plot the results (adapted from plotRVR.m)
close all

figure

plot(newX, newY,':', 'Color', COLOR.sinc);
hold on

plot(x, t, '.', 'Color', COLOR.data);
plot(newX, newY_rvm,'r-','LineWidth', 1, 'Color', COLOR.pred);
plot(x(MODEL.RVs_idx), t(MODEL.RVs_idx),'o', 'Color', COLOR.rv);


area(newX, MODEL.mu_star + MODEL.sigma_star, -1, 'EdgeAlpha', 0, 'FaceColor', 'r', 'FaceAlpha', 0.1);
area(newX, MODEL.mu_star - MODEL.sigma_star, -1, 'EdgeAlpha', 0, 'FaceColor', 'w', 'FaceAlpha', 1.0);
plot(newX, newY_rvm + sqrt(1/MODEL.beta),'r:','LineWidth', 1, 'Color', COLOR.pred);
plot(newX, newY_rvm - sqrt(1/MODEL.beta),'r:','LineWidth', 1, 'Color', COLOR.pred);
   

%redraw
plot(newX, newY,':', 'Color', COLOR.sinc);
plot(x, t, '.', 'Color', COLOR.data);
plot(newX, newY_rvm,'r-','LineWidth', 1, 'Color', COLOR.pred);
plot(x(MODEL.RVs_idx), t(MODEL.RVs_idx),'o', 'Color', COLOR.rv);


xlabel('X', 'Interpreter', 'LaTex');
ylabel('Y', 'Interpreter', 'LaTex');
legend({'Actual Model', 'Datapoints', 'Regression', 'Relevance Vectors'}, 'Location', 'NorthWest', 'Interpreter', 'LaTex')

title_string = sprintf('RVR + RBF Kernel: $\\sigma$ = %g, RV = %d, $\\epsilon_{est}$= %g', MODEL.lengthScale, length(MODEL.RVs_idx), sqrt(1/MODEL.beta)); 
title(title_string, 'Interpreter', 'LaTex');
       