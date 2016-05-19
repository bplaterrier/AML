% NORMALIZE    Normalize data so that it ranges between 1 and 0
%
%       X_ = SB2_KERNELFUNCTION(X)
%
% OUTPUT ARGUMENTS:
%
%       X_       N x D data matrix (normalized)
% 
% INPUT ARGUMENTS:
%
%       X        N x D data matrix
function [X_] = normalize(X)
    X_ = (X - repmat(min(X,[],1),size(X,1),1))*spdiags(1./(max(X,[],1)-min(X,[],1))',0,size(X,2),size(X,2));
end

