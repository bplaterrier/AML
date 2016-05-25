function [ sigma ] = callback( ~, ~, ~, ~, ~, SIGMA, ~, BETA, ~, ~, VARARGIN )
%CALLBACK Summary of this function goes here
%   Detailed explanation goes here
    sigma = SIGMA;
     disp('sigma');
     disp(size(VARARGIN));
         disp('sigmaf');
%     disp(BETA);
   
%     disp(VARARGIN'*SIGMA*VARARGIN);

end

