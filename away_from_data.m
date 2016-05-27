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
COLOR.rv = 'g';       % color of the relevance vectors

%% Plot the data and actual function
plotModel(x, y, t, data, COLOR);

%% Do RVR on the data (rvm_regressor)
close all
% Choose kernel function
OPTIONS.kernel = 'gaussian';        % choose kernel function (see SB2_KernelFunction.m)
OPTIONS.lengthScale = 2.2;          % ~std. dev. for gaussian kernel
OPTIONS.useBias = 0;                % add bias vector to the estimated model
OPTIONS.maxIts = 500;
OPTIONS.BASIS = [];

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

%% Plot the RVR model away from the original data
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

plot(newX, newY,'--', 'Color', COLOR.sinc);
hold on

plot(x, t, '.', 'Color', COLOR.data);
plot(newX, newY_rvm,'r-','LineWidth', 1, 'Color', COLOR.pred);
plot(x(MODEL.RVs_idx), t(MODEL.RVs_idx),'o', 'Color', COLOR.rv);

boundedline(newX,newY_rvm, MODEL.sigma_star, 'r')

% area(newX, MODEL.mu_star + MODEL.sigma_star, -1, 'EdgeAlpha', 0, 'FaceColor', 'r', 'FaceAlpha', 0.1);
% area(newX, MODEL.mu_star - MODEL.sigma_star, -1, 'EdgeAlpha', 0, 'FaceColor', 'w', 'FaceAlpha', 1.0);
plot(newX, newY_rvm + sqrt(1/MODEL.beta),'r:','LineWidth', 1, 'Color', COLOR.pred);
plot(newX, newY_rvm - sqrt(1/MODEL.beta),'r:','LineWidth', 1, 'Color', COLOR.pred);

%redraw
plot(newX, newY,'--', 'Color', COLOR.sinc);
plot(x, t, '.', 'Color', COLOR.data);
plot(newX, newY_rvm,'r-','LineWidth', 1, 'Color', COLOR.pred);
plot(x(MODEL.RVs_idx), t(MODEL.RVs_idx),'o', 'Color', COLOR.rv);

xlabel('X', 'Interpreter', 'LaTex');
ylabel('Y', 'Interpreter', 'LaTex');
legend({'Actual Model', 'Datapoints', 'Regression', 'Relevance Vectors'}, 'Location', 'NorthWest', 'Interpreter', 'LaTex')

title_string = sprintf('RVR + RBF Kernel: $\\sigma$ = %g, RV = %d, $\\epsilon_{est}$= %g', MODEL.lengthScale, length(MODEL.RVs_idx), sqrt(1/MODEL.beta)); 
title(title_string, 'Interpreter', 'LaTex');

%% Do SVR on the data

% Perform a scaling of the data (normalisation)
x_svm = x; %normalize(x);

% Train the model on the data
OPTIONS.svr_type        = 0;            % 0: epsilon-SVR / 1: nu-SVR
OPTIONS.kernel_type     = 2;            % 1: linear / 2: gaussian / 3: polyN / 4: precomputed kernel matrix
OPTIONS.kernel          = 'gaussian';   % kernel type, used for custom basis matrix
OPTIONS.C               = 50;           % penalty factor (default 1)
OPTIONS.nu              = 0.05;         % nu parameter (default 1)
OPTIONS.epsilon         = 0.25;          % epsilon parameter (default 0.1)
OPTIONS.tolerance       = 0.001;        % tolerance of termination criterion (default 0.001)
OPTIONS.lengthScale     = 2.3;          % lengthscale parameter (~std dev for gaussian kernel)
OPTIONS.probabilities   = 0;            % whether to train a SVR model for probability estimates, 0 or 1 (default 0);
OPTIONS.useBias         = 0;            % add bias to the model (for custom basis matrix)

% Train the model
MODEL = svr_train(t, x_svm, OPTIONS);

[y_svm, MODEL] = svm_regressor(x_svm, t, OPTIONS, MODEL);

% Do SVR on data using svmregressor
% [y_svm] = svr_predict(x_svm, MODEL);

%Plot the results
plotSVR(x, y, y_svm, t, data, MODEL, OPTIONS, COLOR);

% Compute errors
mse  = gfit2(t, y_svm, '1');
nmse = gfit2(t, y_svm, '2');
disp([mse nmse]);

%% Plot the SVR model away from the original data
data.N = 300;                   % number of samples
data.scale = 20;                % scale of dimensions

[newX, newY, ~] = generateSinc(data, rseed);

% newX_svm = normalize(newX);

close all

[newY_svm] = svr_predict(newX, MODEL);

% Plot the results (adapted from plotSVR.m)
figure
hold on 
plot(newX, newY,'--', 'Color', COLOR.sinc);
plot(x, t, '.', 'Color', COLOR.data);
plot(newX, newY_svm, '-', 'Color', COLOR.pred)
plot(x(MODEL.sv_indices), t(MODEL.sv_indices),'o', 'Color', COLOR.rv);

boundedline(newX,newY_svm, OPTIONS.epsilon, 'r')

% area(newX, newY_svm + OPTIONS.epsilon, -1, 'EdgeAlpha', 0, 'FaceColor', 'r', 'FaceAlpha', 0.1);
% area(newX, newY_svm - OPTIONS.epsilon, -1, 'EdgeAlpha', 0, 'FaceColor', 'w', 'FaceAlpha', 1);
  
% redraw
plot(newX, newY,'--', 'Color', COLOR.sinc);
plot(x, t, '.', 'Color', COLOR.data);
plot(x, y_svm,'-',  'LineWidth', 1, 'Color', COLOR.pred);
plot(x(MODEL.sv_indices), t(MODEL.sv_indices),'o', 'Color', COLOR.rv);

xlabel('X', 'Interpreter', 'LaTex');
ylabel('Y', 'Interpreter', 'LaTex');
legend({'Actual Model', 'Datapoints', 'Regression', 'Support Vectors', '$\epsilon$-tube'}, 'Interpreter', 'LaTex', 'Location', 'NorthWest')

title_string = sprintf('$\\epsilon$-SVR + RBF kernel: $\\epsilon$ = %g, $\\sigma$ = %g, $C$ =%d, $SV$ = %d', OPTIONS.epsilon, OPTIONS.lengthScale, OPTIONS.C, MODEL.totalSV);
title(title_string, 'Interpreter', 'LaTex');

