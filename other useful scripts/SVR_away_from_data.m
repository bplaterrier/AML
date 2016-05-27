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

%% Plot the regression model away from the original data
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

