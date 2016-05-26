function [y_rvm, mu_star, sigma_star] = rvr_predict(x,  MODEL)
    % RVR_PREDICT % Predicts labels and computes accuracy for new
    % data with a learnt RVM model
    %   input ----------------------------------------------------------------
    %
    %       o X        : (N x D), N  input data points of D dimensionality.
    %       o MODEL    : struct
    %
    %
    %
    %   output ----------------------------------------------------------------
    %
    %       o y_rvm            : (N x 1), decision values
    %
    %
    %% Predict labels given model and data

    % Test
    weights     = MODEL.weights;
    kernel      = MODEL.kernel;
    lengthScale = MODEL.lengthScale;
    useBias     = MODEL.useBias;
    RVs_idx     = MODEL.RVs_idx;
    RVs         = MODEL.RVs;

    % Compute RVM over test data and calculate error
    BASIS	= SB2_KernelFunction(x, RVs, kernel, lengthScale);
    % Add bias vector if necessary
    M = size(BASIS,2);
    if useBias
       BASIS = [BASIS ones(M,1)];
    end
    
    y_rvm	= BASIS*weights;
    
    BASIS	= SB2_KernelFunction(RVs, x, kernel, lengthScale);
    % Compute variable estimated noise variance
    for i=1:length(x),
        mu_star(i) = weights'*BASIS(:,i);
        sigma_star(i) = sqrt(1/MODEL.beta + BASIS(:,i)'*MODEL.SIGMA*BASIS(:,i));
    end
end