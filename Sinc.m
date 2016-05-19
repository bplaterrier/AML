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
data.noise = 0.1;               % scale of noise
data.noiseType = 'gauss';       % type of noise ('gauss' or 'unif')

% Generate sampling space
if data.D==1
    x = [-1:2/(data.N-1):1]'*data.scale;
else
    sqrtN = floor(sqrt(data.N));
    N = sqrtN*sqrtN;
    x = data.scale*[0:sqrtN-1]'/sqrtN;
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
%% Plot settings

COL_sinc = 'k';     % color of the actual function
COL_data = 'b';     % color of the real data
COL_pred = 'r';     % color of the prediction
COL_rv = 'k';       % color of the relevance vectors

%% Plot the data and actual function
figure(1)
whitebg(1,'w')
clf
hold on
if data.D==1,
    plot(x, y,'-', 'Color', COL_sinc);
    plot(x, t, '.', 'Color', COL_data);
else
    mesh(gx, gy, reshape(y,size(gx)), 'edgecolor', COL_sinc, 'facecolor', COL_sinc);
    mesh(gx, gy, reshape(t,size(gx)), 'edgecolor', COL_sinc, 'facecolor' ,COL_data);
end

set(gca,'FontSize',12)
drawnow
%% Do SVR on the data

% Perform a scaling of the data (normalisation)
x_svm = (x - repmat(min(x,[],1),size(x,1),1))*spdiags(1./(max(x,[],1)-min(x,[],1))',0,size(x,2),size(x,2));
y = (y - repmat(min(y,[],1),size(y,1),1))*spdiags(1./(max(y,[],1)-min(y,[],1))',0,size(y,2),size(y,2));
% Choose kernel function
kernel = 'gaussian';        % choose kernel function (see SB2_KernelFunction.m)
lengthScale = 0.2;          % ~std. dev. for gaussian kernel

% Create kernel matrix using data-parameterized kernel functions
K = KernelFunction(x_svm, x_svm, kernel, lengthScale);
K_ = [(1:length(x))', K];   % indices required for computation

% Train the model on the data
TYPE = 3;           % 0: C-SVC / 1: nu-SVC / 2: one-class SVM / 3: epsilon-SVR / 4: nu-SVR (default 2)
KERNEL = 4;         % 4: precomputed kernel matrix
C = 10;             % penalty factor (default 1)
NU = 0.05;             % nu parameter (default 1)
EPSILON = 0.1;      % epsilon parameter (default 0.1)
TOLERANCE = 0.001;  % tolerance of termination criterion (default 0.001)
PROBABILITIES = 0;  % whether to train a SVR model for probability estimates, 0 or 1 (default 0);

OPTIONS = ['-s ' num2str(TYPE) ' -t ' num2str(KERNEL) ' -c ' num2str(C) ' -n ' num2str(NU) ' -p ' num2str(EPSILON) ' -b ' num2str(PROBABILITIES)];
model = svmtrain(t, K_, OPTIONS);

% Useful results ? (see README for more details)
Parameters = model.Parameters;
sv_indices = model.sv_indices;
sv_coef = model.sv_coef;
b = -model.rho;

% Proceed to prediction
[y_svm, ~, ~] = svmpredict(t, K_, model);

if data.D==1,
    plot(x, y_svm,'-','LineWidth', 1, 'Color', COL_pred);
    plot(x(sv_indices), t(sv_indices),'o', 'Color', COL_rv);
else
    mesh(gx, gy, reshape(y_svm,size(gx)), 'edgecolor', COL_sinc, 'facecolor', COL_sinc);
    plot3(x(sv_indices,1), x(sv_indices,2), t(sv_indices), 'Color' ,COL_rv);
end

%% Crossvalidation for parameter selection SVR (example, needs to be adapted)
% Parameter selection
% bestcv = 0;
% for log2c = -1:3,
%   for log2g = -4:1,
%     cmd = ['-v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)]; % -v stands for CV, 5: 5-fold CV
%     cv = svmtrain(heart_scale_label, heart_scale_inst, cmd); % training only returns mean-squared error in CV mode
%     if (cv >= bestcv),
%       bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
%     end
%     fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
%   end
% end

%% Do RVR on the data

% % % Choose kernel function
% kernel = 'gaussian';        % choose kernel function (see SB2_KernelFunction.m)
% lengthScale = 0.2;          % ~std. dev. for gaussian kernel
% useBias = 0;              % add bias vector to the estimated model
% 
% % % Normalize the data
% x_rvm = (x - repmat(min(x,[],1),size(x,1),1))*spdiags(1./(max(x,[],1)-min(x,[],1))',0,size(x,2),size(x,2));
% 
% % Create basis (design) matrix using data-parameterized kernel functions
% BASIS = KernelFunction(x_rvm, x_rvm, kernel, lengthScale);
% M = size(BASIS,2);
% 
% % Add bias vector if necessary
% if useBias
%    BASIS = [BASIS ones(M,1)]; 
% end
% 
% % Set algorithm options
% OPTIONS  = SB2_UserOptions('iterations', 500, 'diagnosticLevel', 2, 'monitor', 10);
% SETTINGS = SB2_ParameterSettings('NoiseStd',0.1);
% 
% % Now run the main SPARSEBAYES function
% [PARAMETER, ~, ~] = SparseBayes('Gaussian', BASIS, t, OPTIONS, SETTINGS);
% 
% % Manipulate the returned weights for convenience later
% w_rvm						= zeros(M,1);
% w_rvm(PARAMETER.Relevant)	= PARAMETER.Value;
% 
% % Compute the inferred prediction function
% y_rvm				= BASIS*w_rvm;
% 
% %% Compute RMS and number of clusters
% rms_model = rms(y - y_rvm)
% 
% %% Plots
% if data.D==1,
%     plot(x, y_rvm,'-','LineWidth', 1, 'Color', COL_pred);
%     plot(x(PARAMETER.Relevant), t(PARAMETER.Relevant),'o', 'Color', COL_rv);
% else
%     mesh(gx, gy, reshape(y_rvm,size(gx)), 'edgecolor', COL_sinc, 'facecolor', COL_sinc);
%     mesh(gx, gy, reshape(t,size(gx)), 'edgecolor', COL_sinc, 'facecolor' ,COL_data);
% end
% hold off
% legend('Actual Model', 'Datapoints', 'Regression', 'Support Vectors', 'Location', 'NorthWest')
