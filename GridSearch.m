close all;
clear all;
clc;

%% Generate the data

% Fix the random seed for reproductibility of results
rseed = 1; % choose a seed
rand('state', rseed);
randn('state', rseed);

% Set default values for data
data.N = 50;                   % number of samples
data.D = 1;                     % dimension of data
data.scale = 10;                % scale of dimensions
data.noise = 0.2;               % scale of noise
data.noiseType = 'gauss';       % type of noise ('gauss' or 'unif')

% Generate sampling space
if data.D==1
    x = (-1:2/(data.N-1):1)'*data.scale;
else
    sqrtN = floor(sqrt(data.N));
    N = sqrtN*sqrtN;
    x = data.scale*(0:sqrtN-1)'/sqrtN;
    [gx, gy]= meshgrid(x);
    x = [gx(:) gy(:)];
end

% Generate latent and target data
if data.D==1,
    y = sin(abs(x))./abs(x);
else
    y = sin(sqrt(sum(x.^2,2)))./sqrt(sum(x.^2,2));
end

switch lower(data.noiseType)
    case 'gauss',
        t = y + randn(size(x,1),1)*data.noise;
    case 'unif',
        t = y + (-data.noise + 2*rand(size(x,1),1)*data.noise);
    otherwise,
        error('Unrecognized noise type: %s', data.noiseType);
end

% Plot settings
colors.sinc = 'k';     % color of the actual function
colors.data = 'b';     % color of the real data
colors.pred = 'r';     % color of the prediction
colors.rv = 'k';       % color of the relevance vectors

%% Plot the data and actual function
figure(1)
whitebg(1,'w')
clf
hold on
if data.D==1,
    plot(x, y,'-', 'Color', colors.sinc);
    plot(x, t, '.', 'Color', colors.data);
else
    mesh(gx, gy, reshape(y,size(gx)), 'edgecolor', colors.sinc, 'facecolor', colors.sinc);
    mesh(gx, gy, reshape(t,size(gx)), 'edgecolor', colors.sinc, 'facecolor' ,colors.data);
end

set(gca,'FontSize',12)
drawnow
%% Do SVR on the data

% Perform a scaling of the data (normalisation)
x_svm = (x - repmat(min(x,[],1),size(x,1),1))*spdiags(1./(max(x,[],1)-min(x,[],1))',0,size(x,2),size(x,2));

% Choose kernel function
svm_opt.kernel = 'gaussian';        % choose kernel function (see SB2_KernelFunction.m)
svm_opt.lengthScale = 0.1;          % ~std. dev. for gaussian kernel

% Create kernel matrix using data-parameterized kernel functions
K = SB2_KernelFunction(x_svm, x_svm, svm_opt.kernel, svm_opt.lengthScale);
K_ = [(1:length(x))', K];   % indices required for computation

% Train the model on the data
svm_opt.TYPE = 3;           % 0: C-SVC / 1: nu-SVC / 2: one-class SVM / 3: epsilon-SVR / 4: nu-SVR (default 2)
svm_opt.KERNEL = 4;         % 4: precomputed kernel matrix
svm_opt.C = 50;            % penalty factor (default 1)
svm_opt.NU = 0.05;          % nu parameter (default 1)
svm_opt.EPSILON = 0.1;      % epsilon parameter (default 0.1)
svm_opt.TOLERANCE = 0.001;  % tolerance of termination criterion (default 0.001)
svm_opt.PROBABILITIES = 0;  % whether to train a SVR model for probability estimates, 0 or 1 (default 0);

svm_opt.string = [...
    ' -s ' num2str(svm_opt.TYPE) ...
    ' -t ' num2str(svm_opt.KERNEL) ...
    ' -c ' num2str(svm_opt.C) ...
    ' -n ' num2str(svm_opt.NU) ...
    ' -p ' num2str(svm_opt.EPSILON) ...
    ' -b ' num2str(svm_opt.PROBABILITIES) ...
    ' -q ' ];

model = svmtrain(t, K_, svm_opt.string);

% Useful results ? (see README for more details)
% Parameters = model.Parameters;
sv_indices = model.sv_indices;
% sv_coef = model.sv_coef;
% a2 = -model.rho;

% Proceed to prediction
[y_svm, ~, ~] = svmpredict(t, K_, model);
figure(1)
if data.D==1,
    plot(x, y_svm,'-','LineWidth', 1, 'Color', colors.pred);
    plot(x(sv_indices), t(sv_indices),'o', 'Color', colors.rv);
else
    mesh(gx, gy, reshape(y_svm,size(gx)), 'edgecolor', colors.sinc, 'facecolor', colors.sinc);
    plot3(x(sv_indices,1), x(sv_indices,2), t(sv_indices), 'Color' ,colors.rv);
end
drawnow
legend('Actual Model', 'Datapoints', 'Regression', 'Support Vectors', 'Location', 'NorthWest')

%% Crossvalidation for parameter selection SVR (example, needs to be adapted)

% Grid search boundaries
Cmin = 1;       Cmax = 100;
Emin = 0.001;    Emax = 0.4;
nsteps = 10;

w = 0.2; % Fixed kernel width
Kgrid = SB2_KernelFunction(x_svm, x_svm, 'gaussian', w);
Kgrid_ = [(1:length(x))', Kgrid];

range_C = linspace(Cmin, Cmax, nsteps);
range_E = linspace(Emin, Emax, nsteps);

errors = zeros(nsteps);
nSV = zeros(nsteps);
r = zeros(nsteps);

% Grid search
for i=1:length(range_C)
    for j=1:length(range_E)
        
        C = range_C(i);
        e = range_E(j);
        
        % Train model
        opt = ['-s 3 -t 4 -c ' num2str(C) ' -p ' num2str(e) ' -q'];
        mGrid = svmtrain(t, Kgrid_, opt);
        
        % Predict & Get performance of the trained model
        [~, error, ~] = svmpredict(t, Kgrid_, mGrid);
        nSV(i,j) = mGrid.totalSV; % Number of support vectors
        errors(i,j) = error(2); % Mean squared error
        r(i,j) = error(3); % Squared correlation coefficient
    end
end

% Plot the results
figure
mesh(range_E, range_C, errors);
xlabel('\epsilon');
ylabel('C');
zlabel('Mean squared error');

figure
mesh(range_E, range_C, r);
xlabel('\epsilon');
ylabel('C');
zlabel('r');
title('Correlation coefficient');
    
figure
mesh(range_E, range_C, nSV);
xlabel('\epsilon');
ylabel('C');
zlabel('N_{SV}');

%% Do RVR on the data

% Choose kernel function
rvm_opt.kernel = 'gaussian';        % choose kernel function (see SB2_KernelFunction.m)
rvm_opt.lengthScale = 0.1;          % ~std. dev. for gaussian kernel
rvm_opt.useBias = 0;                % add bias vector to the estimated model

% Normalize the data
x_rvm = (x - repmat(min(x,[],1),size(x,1),1))*spdiags(1./(max(x,[],1)-min(x,[],1))',0,size(x,2),size(x,2));

% Create basis (design) matrix using data-parameterized kernel functions
rvm_opt.BASIS = SB2_KernelFunction(x_rvm, x_rvm, rvm_opt.kernel, rvm_opt.lengthScale);
M = size(rvm_opt.BASIS,2);

% Add bias vector if necessary
if rvm_opt.useBias
   rvm_opt.BASIS = [rvm_opt.BASIS ones(M,1)]; 
end

% Set algorithm options
rvm_opt.OPTIONS  = SB2_UserOptions('iterations', 500, 'diagnosticLevel', 0, 'monitor', 10);
rvm_opt.SETTINGS = SB2_ParameterSettings('NoiseStd',0.1);

% Now run the main SPARSEBAYES function
[PARAMETER, ~, ~] = SparseBayes('Gaussian', rvm_opt.BASIS, t, rvm_opt.OPTIONS, rvm_opt.SETTINGS);

% Manipulate the returned weights for convenience later
w_rvm						= zeros(M,1);
w_rvm(PARAMETER.Relevant)	= PARAMETER.Value;

% Compute the inferred prediction function
y_rvm				= rvm_opt.BASIS*w_rvm;

figure
if data.D==1,
    plot(x, y,'-', 'Color', colors.sinc);
    hold on
    plot(x, t, '.', 'Color', colors.data);
    plot(x, y_rvm,'LineWidth', 1, 'Color', colors.pred);
  
    plot(x(PARAMETER.Relevant), t(PARAMETER.Relevant),'o', 'Color', colors.rv);
else
    mesh(gx, gy, reshape(y_rvm,size(gx)), 'edgecolor', colors.sinc, 'facecolor', colors.sinc);
    mesh(gx, gy, reshape(t,size(gx)), 'edgecolor', colors.sinc, 'facecolor' ,colors.data);
end
hold off
legend('Actual Model', 'Datapoints', 'Regression', 'Relevance Vectors', 'Location', 'NorthWest')

% %%% Print results of RVR %%%
% fprintf('\n');
% 
% % Mean Squared Error
% rvm_MSE = (1/data.N)*sum((y_rvm - t).^2); 
% fprintf('Mean squared error = %g\n', rvm_MSE);
% 
% % Squared correlation coefficient
% rvm_r2 = (data.N*sum(y_rvm.*t) - sum(y_rvm)*sum(t))^2/(data.N*sum(y_rvm.^2) - sum(y_rvm)^2 * data.N*sum(t.^2) - sum(t)^2);
% fprintf('Squared correlation coefficient = %g\n', rvm_r2);
% 
% % Number of Relevance Vectors
% rvm_nRV = length(PARAMETER.Relevant);
% fprintf('Number of relevant vectors = %d \n', rvm_nRV);

%% RVR Grid Search

