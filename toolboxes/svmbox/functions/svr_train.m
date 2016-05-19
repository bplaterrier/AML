function [MODEL] = svr_train(t, x, OPTIONS)
% SVM_TRAIN Trains an SVM Model using LIBSVM
%
%   input ----------------------------------------------------------------
%
%       o y               : (N x 1), number of output datapoints.
%
%       o X               : (N x D), number of input datapoints.
%
%       o OPTIONS     : struct
%
%
%   output ----------------------------------------------------------------
%
%       o MODEL           : struct
%
%
%% LIBSVM OPTIONS
% options:
% -s svm_type : set type of SVM (default 0)
% 	0 -- C-SVC
% 	1 -- nu-SVC
% 	2 -- one-class SVM
% 	3 -- epsilon-SVR
% 	4 -- nu-SVR
% -t kernel_type : set type of kernel function (default 2)
% 	0 -- linear: u'*v
% 	1 -- polynomial: (gamma*u'*v + coef0)^degree
% 	2 -- radial basis function: exp(-gamma*|u-v|^2)
% 	3 -- sigmoid: tanh(gamma*u'*v + coef0)
% -d degree : set degree in kernel function (default 3)
% -g gamma : set gamma in kernel function (default 1/num_features)
% -r coef0 : set coef0 in kernel function (default 0)
% -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
% -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
% -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
% -m cachesize : set cache memory size in MB (default 100)
% -e epsilon : set tolerance of termination criterion (default 0.001)
% -h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)
% -b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
% -wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)
%
%% Parse SVM Options
% Parsing Parameter Options for SVM variants
switch OPTIONS.svr_type
    
    case 0 % epsilon-SVR

        switch OPTIONS.kernel_type
            case 0 % linear
                options = strcat({'-s 3 '}, {'-t '}, {''}, {num2str(OPTIONS.kernel_type)},{' -c '}, {''}, {num2str(OPTIONS.C)},{' -h 0 -p '}, {''}, {num2str(OPTIONS.epsilon)}, {' -b '}, {''}, {num2str(OPTIONS.probabilities)});
            case 1 % poly
                options = strcat({'-s 3 '}, {'-t '}, {''}, {num2str(OPTIONS.kernel_type)},{' -d '}, {''}, {num2str(OPTIONS.degree)},{' -r '}, {''}, {num2str(OPTIONS.coeff)}, {' -c '}, {''},{num2str(OPTIONS.C)},{' -h 0 -g 1 -p '}, {''}, {num2str(OPTIONS.epsilon)}, {' -b '}, {''}, {num2str(OPTIONS.probabilities)});
            case 2 % gauss
                OPTIONS.gamma = 1 / (2*OPTIONS.lengthScale^2);
                options = strcat({'-s 3 '}, {'-t '}, {''}, {num2str(OPTIONS.kernel_type)},{' -c '}, {''}, {num2str(OPTIONS.C)},{' -g '}, {''}, {num2str(OPTIONS.gamma)},{' -h 0 -p '}, {''}, {num2str(OPTIONS.epsilon)}, {' -b '}, {''}, {num2str(OPTIONS.probabilities)});
        end
        
    case 1 % nu-SVR
   
        switch OPTIONS.kernel_type
            case 0 % linear
                options = strcat({'-s 4 '}, {'-t '}, {''}, {num2str(OPTIONS.kernel_type)},{' -n '}, {''}, {num2str(OPTIONS.nu)},{' -h 0 -c '}, {''}, {num2str(OPTIONS.C)}, {' -b '}, {''}, {num2str(OPTIONS.probabilities)});
            case 1 % gauss
                options = strcat({'-s 4 '}, {'-t '}, {''}, {num2str(OPTIONS.kernel_type)},{' -d '}, {''}, {num2str(OPTIONS.degree)},{' -r '}, {''}, {num2str(OPTIONS.coeff)}, {' -n '}, {''},{num2str(OPTIONS.nu)},{' -h 0 -g 1 -c '}, {''}, {num2str(OPTIONS.C)}, {' -b '}, {''}, {num2str(OPTIONS.probabilities)});
            case 2 % poly
                OPTIONS.gamma = 1 / (2*OPTIONS.lengthScale^2);
                options = strcat({'-s 4 '}, {'-t '}, {''}, {num2str(OPTIONS.kernel_type)},{' -n '}, {''}, {num2str(OPTIONS.nu)},{' -g '}, {''}, {num2str(OPTIONS.gamma)},{' -h 0 -c '}, {''}, {num2str(OPTIONS.C)}, {' -b '}, {''}, {num2str(OPTIONS.probabilities)});
        end
end
        
options = options{1};


%% Train SVM Model
MODEL = svmtrain(t, x, options);

%% The LIBSVM Model struct will contain the following parameters
% -Parameters: parameters
% -nr_class: number of classes; = 2 for regression/one-class svm
% -totalSV: total #SV
% -rho: -b of the decision function(s) wx+b
% -Label: label of each class; empty for regression/one-class SVM
% -sv_indices: values in [1,...,num_traning_data] to indicate SVs in the training set
% -ProbA: pairwise probability information; empty if -b 0 or in one-class SVM
% -ProbB: pairwise probability information; empty if -b 0 or in one-class SVM
% -nSV: number of SVs for each class; empty for regression/one-class SVM
% -sv_coef: coefficients for SVs in decision functions
% -SVs: support vectors

end