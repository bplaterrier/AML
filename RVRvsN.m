close all;
clear all;
clc;

%% Generate the data

% Fix the random seed for reproductibility of results
rseed = 1; % choose a seed
rand('state', rseed);
randn('state', rseed);

% Set default values for data
data.N = 1000;                   % number of samples
data.D = 1;                     % dimension of data
data.scale = 10;                % scale of dimensions
data.noise = 0.1;               % scale of noise
data.noiseType = 'gauss';       % type of noise ('gauss' or 'unif')

% Generate sampling space
x = [-1:2/(data.N-1):1]'*data.scale;

% Generate latent and target data
y = sin(abs(x))./abs(x);
switch lower(data.noiseType)
    case 'gauss',
        t = y + randn(size(x,1),1)*data.noise;
    case 'unif',
        t = y + (-data.noise + 2*rand(size(x,1),1)*data.noise);
    otherwise,
        error('Unrecognized noise type: %s', data.noiseType);
end

% Normalize the data
x_rvm = (x - repmat(min(x,[],1),size(x,1),1))*spdiags(1./(max(x,[],1)-min(x,[],1))',0,size(x,2),size(x,2));

%% Plot settings

%% Coarse grid search for best length scale

% Choose kernel function and parameters
kernel = 'gaussian';       
lengthScale = [0.1 0.2 0.3 0.4 0.5 0.6 0.7];
useBias = 1;

best_model.RMS = Inf;
best_model.index = 0;

display('Performing coarse grid search for best length scale \n');

for i=1:length(lengthScale),   
    % Create basis (design) matrix using data-parameterized kernel functions
    BASIS = KernelFunction(x_rvm, x_rvm, kernel, lengthScale(i));
    M = size(BASIS,2);

    % Add bias vector if necessary
    if useBias
       BASIS = [BASIS ones(M,1)]; 
    end

    % Set algorithm options
    OPTIONS  = SB2_UserOptions('iterations', 500, 'diagnosticLevel', 2, 'monitor', 10);
    SETTINGS = SB2_ParameterSettings('NoiseStd',0.1);

    % Now run the main SPARSEBAYES function
    [PARAMETER, ~, ~] = SparseBayes('Gaussian', BASIS, t, OPTIONS, SETTINGS);

    % Manipulate the returned weights for convenience later
    w_rvm						= zeros(M,1);
    w_rvm(PARAMETER.Relevant)	= PARAMETER.Value;

    % Compute the inferred prediction function
    y_rvm				= BASIS*w_rvm;

    % Compute RMS and choose best lenth scale
    modelRMS = rms(y - y_rvm);
    if modelRMS < best_model.RMS,
        best_model.RMS = modelRMS;
        best_model.index = i;
    end
end

display(['Best length scale around: ' num2str(lengthScale(best_model.index))]);

%% Refined grid search for best length scale

% Choose a narrow grid to search in
lengthScale = [0.16 0.17 0.18 0.19 0.20 0.21 0.22 0.23 0.24];
best_model.RMS = Inf;
best_model.index = 0;

display('Performing refined grid search for best length scale \n');

for i=1:length(lengthScale),   
    % Create basis (design) matrix using data-parameterized kernel functions
    BASIS = KernelFunction(x_rvm, x_rvm, kernel, lengthScale(i));
    M = size(BASIS,2);

    % Add bias vector if necessary
    if useBias
       BASIS = [BASIS ones(M,1)]; 
    end

    % Set algorithm options
    OPTIONS  = SB2_UserOptions('iterations', 500, 'diagnosticLevel', 2, 'monitor', 10);
    SETTINGS = SB2_ParameterSettings('NoiseStd',0.1);

    % Now run the main SPARSEBAYES function
    [PARAMETER, ~, ~] = SparseBayes('Gaussian', BASIS, t, OPTIONS, SETTINGS);

    % Manipulate the returned weights for convenience later
    w_rvm						= zeros(M,1);
    w_rvm(PARAMETER.Relevant)	= PARAMETER.Value;

    % Compute the inferred prediction function
    y_rvm				= BASIS*w_rvm;

    % Compute RMS and choose best lenth scale
    modelRMS = rms(y - y_rvm);
    if modelRMS < best_model.RMS,
        best_model.RMS = modelRMS;
        best_model.index = i;
    end
end

display(['Best refined length scale: ' num2str(lengthScale(best_model.index))]);

%% Compute RMS and number of relevant vectors as a function of N (numer of features)

% % Fix the random seed for reproductibility of results
rseed = 1; % choose a seed
rand('state', rseed);
randn('state', rseed);

% Set default values for data
data.N = [100:10:1000];                % scale of dimensions
data.noise = 0.1;               % scale of noise
data.noiseType = 'gauss';       % type of noise ('gauss' or 'unif')

% Kernel parameters
lengthScale = 0.17;
RMS = zeros(length(data.N),1);
RV_number = zeros(length(data.N));

for i=1:length(data.N),
    % Generate sampling space
    x = [-1:2/(data.N(i)-1):1]'*data.scale;

    % Generate latent and target data
    y = sin(abs(x))./abs(x);
    switch lower(data.noiseType)
        case 'gauss',
            t = y + randn(size(x,1),1)*data.noise;
        case 'unif',
            t = y + (-data.noise + 2*rand(size(x,1),1)*data.noise);
        otherwise,
            error('Unrecognized noise type: %s', data.noiseType);
    end
    
    % Normalize the data
    x_rvm = (x - repmat(min(x,[],1),size(x,1),1))*spdiags(1./(max(x,[],1)-min(x,[],1))',0,size(x,2),size(x,2));
    
    % Create basis (design) matrix using data-parameterized kernel functions
    BASIS = KernelFunction(x_rvm, x_rvm, kernel, lengthScale);
    M = size(BASIS,2);

    % Add bias vector if necessary
    if useBias
       BASIS = [BASIS ones(M,1)]; 
    end

    % Set algorithm options
    OPTIONS  = SB2_UserOptions('iterations', 500, 'diagnosticLevel', 2, 'monitor', 10);
    SETTINGS = SB2_ParameterSettings('NoiseStd',0.1);

    % Now run the main SPARSEBAYES function
    [PARAMETER, ~, ~] = SparseBayes('Gaussian', BASIS, t, OPTIONS, SETTINGS);

    % Manipulate the returned weights for convenience later
    w_rvm						= zeros(M,1);
    w_rvm(PARAMETER.Relevant)	= PARAMETER.Value;

    % Compute the inferred prediction function
    y_rvm				= BASIS*w_rvm;

    % Compute RMS and choose best lenth scale
    RMS(i) = rms(y - y_rvm);
    RV_number(i) = length(PARAMETER.Relevant);
end
figure;
plot(data.N, RMS);
figure;
plot(data.N, RV_number);
