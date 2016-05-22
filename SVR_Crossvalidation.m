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

%% Do SVR on the data

% Perform a scaling of the data (normalisation)
x_svm = normalize(x);

% Train the model on the data
OPTIONS.svr_type        = 0;            % 0: epsilon-SVR / 1: nu-SVR
OPTIONS.kernel_type     = 2;            % 1: linear / 2: gaussian / 3: polyN / 4: precomputed kernel matrix
OPTIONS.kernel          = 'gaussian';   % kernel type, used for custom basis matrix
OPTIONS.C               = 50;           % penalty factor (default 1)
OPTIONS.nu              = 0.05;         % nu parameter (default 1)
OPTIONS.epsilon         = 0.1;          % epsilon parameter (default 0.1)
OPTIONS.tolerance       = 0.001;        % tolerance of termination criterion (default 0.001)
OPTIONS.lengthScale     = 0.2;          % lengthscale parameter (~std dev for gaussian kernel)
OPTIONS.probabilities   = 0;            % whether to train a SVR model for probability estimates, 0 or 1 (default 0);
OPTIONS.useBias         = 0;            % add bias to the model (for custom basis matrix)

% Train the model
MODEL = svr_train(t, x_svm, OPTIONS);

% Do SVR on data using svmregressor
[y_svm] = svr_predict(x_svm, MODEL);

%Plot the results
plotSVR(x, y, y_svm, t, data, MODEL, OPTIONS, COLOR)

% Compute errors
mse  = gfit2(t, y_svm, '1');
nmse = gfit2(t, y_svm, '2');
disp([mse nmse]);

%% Do K-fold crossvalidation on the data

disp('Parameter grid search SVR');

limits_C        = [1 500];   % Limits of penalty C
limits_epsilon  = [0.05 1];  % Limits of epsilon
limits_w        = [0.25 2];  % Limits of kernel width \sigma
parameters      = vertcat(limits_C, limits_epsilon, limits_w);
step            = 5;         % Step of parameter grid 
Kfold           = 10;

metric = 'nmse';

% Do Grid Search
[ ctest, ctrain , cranges ] = ml_grid_search_regr( x_svm(:), t(:), Kfold, parameters, step);



%% Get CV statistics
statsSVR = ml_get_cv_grid_states_regression(ctest,ctrain);

% Plot Heatmaps from Grid Search 
cv_plot_options              = [];
cv_plot_options.title        = strcat('$\epsilon$-SVR :: ', num2str(Kfold),'-fold CV with RBF');
cv_plot_options.para_names  = {'C','\epsilon', '\sigma'};
cv_plot_options.metrics      = {'nmse'};
cv_plot_options.param_ranges = [cranges{1} ; cranges{2}; cranges{3}];

parameters_used = [cranges{1};cranges{2};cranges{3}];

if exist('hcv','var') && isvalid(hcv), delete(hcv);end
hcv = ml_plot_cv_grid_states_regression(statsSVR,parameters_used,cv_plot_options);


%% Get optimal parameters and plot result
[min_metricSVR,indSVR] = min(statsSVR.test.(metric).mean(:));
[C_min, eps_min, sigma_min] = ind2sub(size(statsSVR.test.(metric).mean),indSVR);
C_opt = cranges{1}(C_min);
epsilon_opt = cranges{2}(eps_min);
sigma_opt = cranges{3}(sigma_min);

% Test model with optimal parameter
% clear svr_options
OPTIONS.C           = C_opt;   % set the parameter C of C-SVC, epsilon-SVR, and nu-SVR 
OPTIONS.epsilon     = epsilon_opt;  % nu \in (0,1) (upper-bound for misclassifications on margni and lower-bound for # of SV) for nu-SVM
OPTIONS.lengthScale = sigma_opt;   %  radial basis function: exp(-gamma*|u-v|^2), gamma = 1/(2*sigma^2)

[YSVR, modelSVR] = svm_regressor(x, y, OPTIONS, []);


% Plot SVR Regressive function, support vectors and epsilon tube
plotSVR(x, y, YSVR, t, data, MODEL, OPTIONS, COLOR)
% plotSVR( X, y, YSVR);
