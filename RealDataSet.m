close all;
clear;
clc;

%% Load the real data
load housing.mat

t = MEDV;
x = [CRIM ZN INDUS CHAS NOX RM AGE DIS RAD TAX PTRATIO B LSTAT];

x = normalize(x);

%% RVR

% Choose kernel function
OPTIONS.kernel = 'gaussian';        % choose kernel function (see SB2_KernelFunction.m)
OPTIONS.lengthScale = 0.1;          % ~std. dev. for gaussian kernel
OPTIONS.useBias = 0;                % add bias vector to the estimated model
OPTIONS.maxIts = 500;
OPTIONS.BASIS = [];

MODEL = [];
[y_rvm, MODEL] = rvm_regressor(x, t, OPTIONS, MODEL);

% Train the model
% [MODEL] = rvr_train(x, t, OPTIONS);

% Predict values
% [y_rvm, mu_star, sigma_star] = rvr_predict(x, MODEL);

gfit2(t, y_rvm, '2')

[sortedX, index] = sort(x(:,3));
plot(sortedX, y_rvm(index))

%% RVR crossvalidation 
%% K-fold cross validation 

Kfold = 10;
disp('Parameter grid search RVR');

%Set RVR OPTIONS%
OPTIONS.useBias = 0;
OPTIONS.maxIts  = 100;
OPTIONS.BASIS = [];
OPTIONS.kernel = 'gaussian';
OPTIONS.kernel_ = 'gauss';

rbf_vars = [0.1:0.1:20];

test  = cell(length(rbf_vars),1);
train = cell(length(rbf_vars),1);
 
tstart = tic;
for i=1:length(rbf_vars)
    disp(['[' num2str(i) '/' num2str(length(rbf_vars)) ']']);
    
    OPTIONS.lengthScale     = rbf_vars(i);   %  radial basis function: exp(-gamma*|u-v|^2), gamma = 1/(2*sigma^2)    
    
    f                       = @(X,y,model)rvm_regressor(X,y,OPTIONS,model);
    [test_eval,train_eval]  = ml_kcv(x,t,Kfold,f,'regression');
    
    
    test{i}                 = test_eval;
    train{i}                = train_eval;
    disp(' ');
end
time = toc(tstart);
disp(time)

% Get Statistics
[ stats ] = ml_get_cv_grid_states_regression(test,train);

% Plot Statistics

options             = [];
options.title       = 'RVR k-CV';
options.metrics     = {'nmse'};     % <- you can add many other metrics, see list in next cell box
options.para_name   = 'variance rbf';

[handle,handle_test,handle_train] = ml_plot_cv_grid_states_regression(stats,rbf_vars,options);

%% Optimal parameters
[min_metricSVR,indSVR] = min(stats.test.('nmse').mean(:));
[sigma_min] = ind2sub(size(stats.test.('nmse').mean),indSVR);


sigma_opt = rbf_vars(sigma_min); % A garder


OPTIONS.lengthScale   = sigma_opt;
clear model
[y_rvm, MODEL] = rvm_regressor(x,t,OPTIONS,[]);

% A garder
gfit2(t, y_rvm, '2')
length(MODEL.RVs_idx)

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

%% SVR Cross-validation
%% Do K-fold crossvalidation on the data

disp('Parameter grid search SVR');

limits_C        = [1 500];  % Limits of penalty C
limits_epsilon  = [0.5 2];  % Limits of epsilon
limits_w        = [0.1 1];  % Limits of kernel width \sigma
parameters      = vertcat(limits_C, limits_epsilon, limits_w);
step            = 2;         % Step of parameter grid 
Kfold           = 10;

metric = 'nmse';

tstart = tic;
% Do Grid Search
[ ctest, ctrain , cranges ] = ml_grid_search_regr( x, t, Kfold, parameters, step);


time = toc(tstart);
disp(time);

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

% a garder
C_opt = cranges{1}(C_min);
epsilon_opt = cranges{2}(eps_min);
sigma_opt = cranges{3}(sigma_min);

% Test model with optimal parameter
% clear svr_options
OPTIONS.C           = C_opt;   % set the parameter C of C-SVC, epsilon-SVR, and nu-SVR 
OPTIONS.epsilon     = epsilon_opt;  % nu \in (0,1) (upper-bound for misclassifications on margni and lower-bound for # of SV) for nu-SVM
OPTIONS.lengthScale = sigma_opt;   %  radial basis function: exp(-gamma*|u-v|^2), gamma = 1/(2*sigma^2)

MODEL = svr_train(t, x, OPTIONS);
[y_svm] = svr_predict(x, MODEL);

% a garder
MODEL.totalSV
gfit2(t, y_svm, '2')