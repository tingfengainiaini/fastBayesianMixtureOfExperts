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

clear ;

load SData;

%% Create BME
BME = BMECreate('NumExperts', 3 , 'MaxIt', 20, 'EType', 'frvm', 'ENbf', 0.1, 'EKernel', 'rbf', 'EKParam', 0.5, ...
    'GType', 'smlr', 'GNbf', 0.1, 'GKernel', 'rbf', 'GKParam', 0.5);
%% Initialize BME using kmeans clustering
BME = BMEInit(BME, Input, Target, Target, Input) ; 

tic;
%% Now run the EM Algorithm 
BME = BMETrain(BME, Target, Target) ;  
toc;

%% Display results 
NumInput = size(Input,1); 
DataPointColors = {'r.','g.','b.','k.','m.'} ; 
LineColors = {'r-','g-','b-','k-','m-'} ; 

%%------------------------------------------------------------------
%Clustering of the training data using Posterior and corresponding experts
h1 = figure ; 
hold on ; 
for i =1:NumInput     
    [MaxVal MaxI] = max(BME.Gatings.Posteriors(i,:));     
    plot(Input(i,1),Target(i,1),DataPointColors{MaxI});     
end

for i = 1:BME.NumExperts   
   plot(Input,BME.Experts.Means(:,i),LineColors{i});  
end
hold off ;


%%------------------------------------------------------------------
%Gate Distribution 
    
h2 = figure ; 
hold on ; 
for i =1:NumInput     
    [MaxVal MaxI] = max(BME.Gatings.Outputs(i,:));     
    plot(Input(i,1), Target(i,1), DataPointColors{MaxI});     
end

MinInput = min(Input(:,1)) ; 
MaxInput = max(Input(:,1)) ; 
NumPts = 100 ; 
GateInput = MinInput:(MaxInput - MinInput)/(NumPts - 1):MaxInput ; 
GateInput = GateInput';
if strcmpi(BME.Gatings.Kernel, 'linear')
    K = GateInput;
else
    K = EvalKernel(GateInput, Input, BME.Gatings.Kernel, BME.Gatings.KParam);
end
BME.Gatings.Outputs  =  exp([ones(size(GateInput,1),1) K]*BME.Gatings.Weights);
BME.Gatings.Outputs = BMEGatingsOutputsNorm(BME);
for i = 1:BME.NumExperts   
   plot(GateInput, BME.Gatings.Outputs(:,i), LineColors{i});  
end
hold off ;
