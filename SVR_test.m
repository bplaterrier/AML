close all;
clear all;
clc;

%% Generate the data

% Fix the random seed for reproductibility of results
rseed = 1; % choose a seed
rand('state', rseed);
randn('state', rseed);

% Set default values for data
data.N = 200;                   % number of samples
data.D =    1;                     % dimension of data
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
% Plot settings

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

% Proceed to prediction (using toolbox function)
[y_svm, errors, ~] = svmpredict(t, x_svm, MODEL);

%% Do SVR on data using svmregressor
[y_svm] = svm_regressor(x_svm, t, OPTIONS, MODEL);

%% Plot the results
figure(1)
if data.D==1,
    plot(x, y_svm,'-','LineWidth', 1, 'Color', COL_pred);
    plot(x(MODEL.sv_indices), t(MODEL.sv_indices),'o', 'Color', COL_rv);
else
    mesh(gx, gy, reshape(y_svm,size(gx)), 'edgecolor', COL_sinc, 'facecolor', COL_sinc);
    plot3(x(MODEL.sv_indices,1), x(MODEL.sv_indices,2), t(MODEL.sv_indices), 'Color' ,COL_rv);
end
drawnow
legend('Actual Model', 'Datapoints', 'Regression', 'Support Vectors', 'Location', 'NorthWest')
