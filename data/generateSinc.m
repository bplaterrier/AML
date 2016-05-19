% GENERATESINC    Normalize data so that it ranges between 1 and 0
%
%       X_ = GENERATESINC(X)
%
% OUTPUT ARGUMENTS:
%
%       Y        N x D data matrix corresponding to latent variable
%       T        N x D data matrix corresponding to (noisy) observation
% 
% INPUT ARGUMENTS:
%
%       X        N x D data matrix
%       noise    string specifying type of noise ('gauss' or 'unif')
%       scale    scale of the noise
%       
function [Y, T] = generateSinc(X, noise, scale)

    [N,D] = size(X);
    
    if D==1,
        Y = sin(abs(X))./abs(X);
    else
        Y = sin(sqrt(sum(X.^2,2)))./sqrt(sum(X.^2,2));
    end
    
    switch lower(noise)
        case 'gaussian',
            T = Y + scale*randn(N,1);
        case 'unif',
            T = Y + (-scale + 2*scale*rand(N,1));
        otherwise,
            error('Unrecognized noise type: %s', noise);
    end
end

