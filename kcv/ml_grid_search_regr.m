function [ ctest, ctrain, ranges ] = ml_grid_search_regr( x, y, Kfold, parameters, step)
%ML_GRID_SEARCH_CLASS DO grid_search for regression

% parameters is 1x3 vector with limits (start, end) and step


nbParameters = size(parameters,1);
ranges = cell(nbParameters,1);

for i = 1:nbParameters
    ranges{i} = linspace(parameters(i,1), parameters(i,2), step);    
end

steps = allcomb(ranges{1:end});

ctest  = cell(size(steps,1),1);
ctrain = cell(size(steps,1),1);

for i=1:size(steps,1)
    
    svr_options.svr_type        = 0;            % 0: epsilon-SVR / 1: nu-SVR
    svr_options.kernel_type     = 2;            % 1: linear / 2: gaussian / 3: polyN / 4: precomputed kernel matrix
    svr_options.kernel          = 'gaussian';   % kernel type, used for custom basis matrix
    
    svr_options.C               = steps(i,1);
    svr_options.epsilon         = steps(i,2);
    svr_options.kernel_type     = 2;
    svr_options.lengthScale     = steps(i,3);
    svr_options.probabilities   = 0;

    f                           = @(X,y,model)svm_regressor(X,y,svr_options,model);
    [test_eval,train_eval]      = ml_kcv(x,y,Kfold,f,'regression');

    ctest{i}  = test_eval;
    ctrain{i} = train_eval;
    
end

reshapeSize = step*ones(1,nbParameters);

ctest = reshape(ctest,reshapeSize);
ctest = permute(ctest,nbParameters:-1:1);
ctrain = reshape(ctrain,reshapeSize);
ctrain = permute(ctrain,nbParameters:-1:1);

end

