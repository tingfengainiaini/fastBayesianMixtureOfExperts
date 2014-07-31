%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Authors: Liefeng Bo, Cristian Sminchisescu                         %                                         
% Date: 01/12/2010                                                   %
%                                                                    % 
% Copyright (c) 2010  L. Bo, C. Sminchisescu - All rights reserved   %
%                                                                    %
% This software is free for non-commercial usage only. It must       %
% not be distributed without prior permission of the author.         %
% The author is not responsible for implications from the            %
% use of this software. You can run it at your own risk.             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function  BME = BMEInit(BME, Input, Target, ClusterTarget, TestInput)

%% Initialize BME
% Input parameters
%   BME: structured data
%   Input: n x d matrix, each row denotes one sample in training set
%   Target: n x 1 matrix, each entity denotes target value of one sample
%   CluseterTarget: n x t matrix, each row denotes one target vector
%   TestInput: m x d matrix, each row denotes one sample in test set
% Output parameter
%   BME: structured data

%%
BME.Test.NumExperts = BME.NumExperts;
N = size(Input,1);
if nargin > 4
    if strcmpi(BME.Experts.Kernel, 'linear')
        BME.Test.EInput = [ones(size(TestInput,1),1) TestInput];
    else
        K = EvalKernel(TestInput, Input, BME.Experts.Kernel, BME.Experts.KParam);
        BME.Test.EInput = [ones(size(K,1),1) K];
        clear K;
    end
    if strcmpi(BME.Gatings.Kernel, 'linear')
        BME.Test.GInput = [ones(size(TestInput,1),1) TestInput];
    else
        K = EvalKernel(TestInput, Input, BME.Gatings.Kernel, BME.Gatings.KParam);
        BME.Test.GInput = [ones(size(K,1),1) K];
        clear K;
    end
end

%% Initialize experts 
switch lower(BME.Experts.Type)
    case 'rr'
        if strcmpi(BME.Experts.Kernel, 'linear')
            BME.Experts.Input = [ones(N,1) Input];
        else
            K = EvalKernel(Input, Input, BME.Experts.Kernel, BME.Experts.KParam);
            BME.Experts.Input = [ones(N,1) K];
            clear K;
        end
        ED = size(BME.Experts.Input,2);
        DHessian = sum(BME.Experts.Input.^2,2);
        BME.Experts.Alpha = 1e-2;
        [IDInput, Centers] = kMeansCluster(Input, ClusterTarget/10, BME.NumExperts);
    case 'rvm'
        if strcmpi(BME.Experts.Kernel, 'linear')
            BME.Experts.Input = [ones(N,1) Input];
        else
            K = EvalKernel(Input, Input, BME.Experts.Kernel, BME.Experts.KParam);
            BME.Experts.Input = [ones(N,1) K];
            clear K;
        end
        ED = size(BME.Experts.Input,2);
        BME.Experts.Alpha = 1e-5;
        BME.Experts.Weights = zeros(ED,BME.NumExperts);
        [IDInput, Centers] = kMeansCluster(Input, ClusterTarget/10, BME.NumExperts) ; 
        for i = 1:BME.NumExperts
            Indices = find(IDInput == i) ;
            BME.Experts.Beta(1,i) = 1/(var(Target(Indices,:)/10));
            BME.Experts.Variances(i) = var(Target(Indices,:))/10;
        end
    case 'frvm'
        if strcmpi(BME.Experts.Kernel, 'linear')
            BME.Experts.Input = [ones(N,1) Input];
        else
            K = EvalKernel(Input, Input, BME.Experts.Kernel, BME.Experts.KParam);
            BME.Experts.Input = [ones(N,1) K];
            clear K;
        end 
        ED = size(BME.Experts.Input,2);
        [IDInput, Centers] = kMeansCluster(Input, ClusterTarget/10, BME.NumExperts) ; 
        for j = 1:size(Target,2)
            for i = 1:BME.NumExperts
                Indices = find(IDInput == i) ;
                BME.Experts.Variances(j,i) = var(Target(Indices,j))/10;
            end
        end
    otherwise
        disp( ['Unknown method: ' lower(BME.Experts.Type)]);
end

%% Initialize the gatings
switch lower(BME.Gatings.Type)
    case 'mlr'
        if strcmpi(BME.Gatings.Kernel, 'linear')
            BME.Gatings.Input = [ones(N,1) Input];
        else
            K = EvalKernel(Input, Input, BME.Gatings.Kernel, BME.Gatings.KParam);
            BME.Gatings.Input = [ones(N,1) K];
            clear K;
        end
        GD = size(BME.Gatings.Input,2);
        Hessian = BME.Gatings.Input'*BME.Gatings.Input;
        BME.Gatings.Alpha = 1e-2;     
        BME.Gatings.InvH = inv(Hessian + BME.Gatings.Alpha*eye(GD));
        BME.Gatings.InvHH = BME.Gatings.InvH*Hessian;
        BME.Gatings.Weights = zeros(GD,BME.NumExperts);
        BME.Gatings.Outputs = exp(BME.Gatings.Input*BME.Gatings.Weights);
        MinPosterior = 0.01;
        BME.Gatings.Posteriors = MinPosterior*ones(N,BME.NumExperts) ;
        for i = 1:BME.NumExperts
            Indices = find(IDInput == i) ;
            BME.Gatings.Posteriors(Indices,i) = 1 - (BME.NumExperts-1)*MinPosterior;
        end
    case 'smlr'
        if strcmpi(BME.Gatings.Kernel, 'linear')
            BME.Gatings.Input = [ones(N,1) Input];
        else
            K = EvalKernel(Input, Input, BME.Gatings.Kernel, BME.Gatings.KParam);
            BME.Gatings.Input = [ones(N,1) K];
            clear K;
        end
        GD = size(BME.Gatings.Input,2);
        BME.Gatings.Weights = zeros(GD,BME.NumExperts);
        BME.Gatings.Outputs = exp(BME.Gatings.Input*BME.Gatings.Weights);
        MinPosterior = 0.01;
        BME.Gatings.Posteriors = MinPosterior*ones(N,BME.NumExperts) ;
        for i = 1:BME.NumExperts
            Indices = find(IDInput == i) ;
            BME.Gatings.Posteriors(Indices,i) = 1 - (BME.NumExperts-1)*MinPosterior;
        end
    otherwise
        disp( ['Unknown method: ' lower(BME.Gatings.Type)]);
end
