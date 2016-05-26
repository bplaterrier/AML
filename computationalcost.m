clear all;
close all;

%%  computational cost
% Set parameters for computational cost
nb_try = 3;
n_jump = 100;

% Set default values for data
n_limites = [100, 4000];        % number of samples
data.D =    1;                  % dimension of data
data.scale = 10;                % scale of dimensions
data.noise = 0.1;               % scale of noise
data.noiseType = 'gauss';       % type of noise ('gauss' or 'unif')

clear svr_options
% SVR OPTIONS
svr_options.svr_type    = 0;    % 0: epsilon-SVR, 1: nu-SVR
svr_options.C           = 1;    % set the parameter C of C-SVC, epsilon-SVR, and nu-SVR 
svr_options.epsilon     = 0.1;  % set the epsilon in loss function of epsilon-SVR 
svr_options.kernel_type = 2;    % 0: linear: u'*v, 1: polynomial: (gamma*u'*v + coef0)^degree, 2: radial basis function: exp(-gamma*|u-v|^2)
svr_options.sigma       = 0.50; % radial basis function: exp(-gamma*|u-v|^2), gamma = 1/(2*sigma^2)
svr_options.lengthScale = 0.2;  % lengthscale parameter (~std dev for gaussian kernel)
svr_options.probabilities   = 0;    % whether to train a SVR model for probability estimates, 0 or 1 (default 0);
svr_options.useBias         = 0;    % add bias to the model (for custom basis matrix)

clear rvr_options
%Set RVR OPTIONS%
rvr_options.useBias = 0;
rvr_options.maxIts  = 500;
rvr_options.kernel  = 'gaussian';
rvr_options.lengthScale = 0.2;

%computational cost
gf_svr = [];
gf_rvr = [];
time_svr = [];
time_rvr = [];
for k = 1: 1 : nb_try 
    g_svr = [];
    g_rvr = [];
    t_svr = [];
    t_rvr = [];
    for n = n_limites(1): n_jump : n_limites(2)
               
        % Generate True function and data
        data.N = n;
        [x, y_true, y] = generateSinc(data, k);
        x_svr = (x - repmat(min(x,[],1),size(x,1),1))*spdiags(1./(max(x,[],1)-min(x,[],1))',0,size(x,2),size(x,2));
        x_rvr = x_svr;
%         plot (x_svr, y, '.');
             
        % Train SVR Model
        clear model y_rvm
        tstart = tic;
        [y_svr, model] = svm_regressor(x_svr, y, svr_options, []);
        t_svr = [t_svr, toc(tstart)];     
%         %plot
%         plotSVR( x_svr, y_true, y_svr, y, data, model, svr_options, [] );
        
        %goodness of fit
        g_svr = [g_svr, gfit2(y,y_svr,'2')];
        

        %Train RVR Model
        clear model y_rvr
        tstart = tic;
        [model] = rvr_train(x_rvr, y, rvr_options);
        [y_rvr] = rvr_predict(x_rvr,  model);
        t_rvr = [t_rvr, toc(tstart)];
%         %plot
%         plotRVR(x_rvr, y_true, y_rvr, y, data, model, [])

        %goodness of fit
        g_rvr = [g_rvr, gfit2(y,y_rvr,'2')];
        
    end
    gf_svr = [gf_svr; g_svr];
    gf_rvr = [gf_rvr; g_rvr];
    time_svr = [time_svr; t_svr];
    time_rvr = [time_rvr; t_rvr];   
end

% std
nb_samples = n_limites(1): n_jump : n_limites(2);
figure(1)
hold on
errorbar(nb_samples ,mean(time_svr,1),std (time_svr, 0, 1))
errorbar(nb_samples ,mean(time_rvr,1),std (time_rvr, 0, 1))
hold off
figure(2)
hold on
errorbar(nb_samples ,mean(gf_svr,1),std (gf_svr, 0, 1))
errorbar(nb_samples ,mean(gf_rvr,1),std (gf_rvr, 0, 1))
hold off


