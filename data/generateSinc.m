% GENERATESINC    Normalize data so that it ranges between 1 and 0
%
%       X_ = GENERATESINC(X)
%
% OUTPUT ARGUMENTS:
%
%       y        N x D data matrix corresponding to latent variable
%       t        N x D data matrix corresponding to (noisy) observation
% 
% INPUT ARGUMENTS:
%
%       data     structure containing major data attributes
%       rseed    Fix the random seed for reproductibility of results
%       
function [x, y, t] = generateSinc(data, rseed)

    rand('state', rseed);
    randn('state', rseed);
    
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
end

