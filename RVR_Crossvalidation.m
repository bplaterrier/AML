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

plotRVR(x,y,y_rvm,t,data,MODEL,COLOR);

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

% Compute errors
mse  = gfit2(t, y_rvm, '1');
nmse = gfit2(t, y_rvm, '2');
disp([mse nmse]);

%% K-fold cross validation 

Kfold = 10;

disp('Parameter grid search RVR');

%Set RVR OPTIONS%
OPTIONS.useBias = 0;
OPTIONS.maxIts  = 100;

%Set Kernel OPTIONS%
OPTIONS.kernel_ = 'gauss';

rbf_vars = [0.05:0.05:0.5];

test  = cell(length(rbf_vars),1);
train = cell(length(rbf_vars),1);

for i=1:length(rbf_vars)
    disp(['[' num2str(i) '/' num2str(length(rbf_vars)) ']']);
    
    OPTIONS.width       = rbf_vars(i);   %  radial basis function: exp(-gamma*|u-v|^2), gamma = 1/(2*sigma^2)    
    
    f                       = @(X,y,model)rvm_regressor(X,y,OPTIONS,model);
    [test_eval,train_eval]  = ml_kcv(x,y,Kfold,f,'regression');
    
    
    test{i}                 = test_eval;
    train{i}                = train_eval;
    disp(' ');
end


%% Get Statistics

[ stats ] = ml_get_cv_grid_states_regression(test,train);

% Plot Statistics

options             = [];
options.title       = 'RVR k-CV';
options.metrics     = {'nmse'};     % <- you can add many other metrics, see list in next cell box
options.para_name   = 'variance rbf';

[handle,handle_test,handle_train] = ml_plot_cv_grid_states_regression(stats,rbf_vars,options);


%% Get optimal parameters and plot result
[min_metricSVR,indSVR] = min(stats.test.('nmse').mean(:));
[sigma_min] = ind2sub(size(stats.test.('nmse').mean),indSVR);
sigma_opt = rbf_vars(sigma_min);

clear rvr_options

%Set RVR OPTIONS%
OPTIONS.useBias = 0;
OPTIONS.maxIts  = 100;

%Set Kernel OPTIONS%
OPTIONS.kernel_ = 'gauss';
OPTIONS.width   = sigma_opt;

% Train RVR Model
clear model
[y_rvm, MODEL] = rvm_regressor(x,y,OPTIONS,[]);

% Plot RVR function 
plotRVR(x, y, y_rvm, t, data, MODEL, COLOR);


