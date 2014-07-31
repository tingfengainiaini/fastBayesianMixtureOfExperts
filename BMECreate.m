%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Authors: Liefeng Bo, Cristian Sminchisescu                         %                                         
% Date: 01/12/2010                                                   %
%                                                                    % 
% Copyright (c) 2010  Liefeng Bo - All rights reserved               %
%                                                                    %
% This software is free for non-commercial usage only. It must       %
% not be distributed without prior permission of the author.         %
% The author is not responsible for implications from the            %
% use of this software. You can run it at your own risk.             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function BME = BMECreate(varargin)

%% BME create/alter Bayesian mixture of experts options struction.
% Input Parameters
%   NumExperts: number of experts [ positive scalar ]
%   MaxIt: maximum number of iterations allowed [ positive scalar ]
%   MinLogLikeChange: if the change of loglikelihood is smaller than it, stop BME [positive scalar]
%   EType: type of expert [ 'rr', 'rvm' or 'frvm']
%   ENbf: selected input dimension / full input dimension for experts [empty if 'rr' and 'rvm'; (0,1] if frvm]
%   EKernel: type of expert kernel [ 'linear' or 'rbf' ]
%   EKParam: parameter of expert kernel [ positive scalar]
%   GType: type of gatings [ 'mlr'or 'smlr']
%   GNbf: selected input dimension / full input dimension for gatings [empty if 'mlr'; (0,1] if smlr]
%   GKernel: type of expert kernel [ 'linear' or 'rbf' ]
%   GKParam: parameter of expert kernel [ positive scalar]

%% default field
BME.NumExperts = 10;
BME.MaxIt = 10;
BME.MinLogLikeChange = 1e-5;
BME.Experts.Type = 'frvm';
BME.Experts.Nbf = 1;
BME.Experts.Kernel = 'linear';
BME.Experts.KParam = [];
BME.Gatings.Type = 'mlr';
BME.Gatings.Nbf = 1;
BME.Gatings.Kernel = 'linear';
BME.Gatings.KParam = [];

%% set field accroding to data user provides
for i = 1:length(varargin)/2
    Name = varargin{2*(i-1)+1};
    Value = varargin{2*i};
    switch Name
        case {'NumExperts', 'MaxIt'}
            BME = setfield(BME, Name, Value);
        case {'EType', 'ENbf', 'EKernel', 'EKParam'}
            BME.Experts = setfield(BME.Experts, Name(2:end), Value);
        case {'GType', 'GNbf', 'GKernel', 'GKParam'}
            BME.Gatings = setfield(BME.Gatings, Name(2:end), Value);
        otherwise
            disp( ['unknown name: ' Name]);
    end
end