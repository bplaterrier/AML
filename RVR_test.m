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

%% Do RVR on the data

% Choose kernel function
OPTIONS.kernel = 'gaussian';        % choose kernel function (see SB2_KernelFunction.m)
OPTIONS.lengthScale = 0.22;          % ~std. dev. for gaussian kernel
OPTIONS.useBias = 0;                % add bias vector to the estimated model
OPTIONS.maxIts = 500;

% Normalize the data
x_rvm = (x - repmat(min(x,[],1),size(x,1),1))*spdiags(1./(max(x,[],1)-min(x,[],1))',0,size(x,2),size(x,2));

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
x_rvm = (x - repmat(min(x,[],1),size(x,1),1))*spdiags(1./(max(x,[],1)-min(x,[],1))',0,size(x,2),size(x,2));

% Train and predict
MODEL = [];
[y_rvm, MODEL] = rvm_regressor(x, t, OPTIONS, MODEL);