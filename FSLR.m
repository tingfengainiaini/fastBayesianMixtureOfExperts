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

function [Weight, sv] = FSLR(Input, Target, SumOtherGatingsOutputs, nbf)
%% Forward feature selection by gradient boositng and bound optimization 

lambda = 1e-5;
[n, d] = size(Input);
R = [];
sv = [];
nonbf = 1:d;
Grad = (1./(SumOtherGatingsOutputs + 1) - Target);

for i = 1:nbf
    % select the basis function
    score = Input'*Grad;
    score = score(nonbf);
    [temp,tindex] = max(abs(score));
    index = nonbf(tindex);
    nonbf(tindex) = [];
        
%     update the inverse matrix   
    if i == 1
        x = Input(:,index);
        R = 1/(x'*x + lambda); % compute the inverse matrix
        sv = [sv,index];
        SvInput = Input(:,sv);
        Weight = SubProblem(SvInput, Target, SumOtherGatingsOutputs, R, 0);
    else
        x = Input(:,index);
        h = Input'*x; h = h(sv);
        beta = R*h;
        gamma = (x'*x + lambda - h'*beta);
        R = [R zeros(i-1,1); zeros(1,i-1) 0] + [beta; -1]*[beta' -1]./gamma;
        sv = [sv,index];
        SvInput = Input(:,sv);
        Weight = SubProblem(SvInput, Target, SumOtherGatingsOutputs, R, [Weight; 0]);
    end
    % print the result
    GatingOutput = exp(SvInput*Weight);
    Grad = (GatingOutput./(SumOtherGatingsOutputs + GatingOutput) - Target);
%     Error = sum(Target.*(SvInput*Weight) - log(GatingOutput + SumOtherGatingsOutputs));
%     disp(['Error: ' num2str(Error)]);
end

function Weight = SubProblem(Input, Target, SumOtherGatingsOutputs,InvH, Weight)
% Optimize based on bound optimization
MaxIts = 20;
MinWeightChange = 1e-3;
count = 0;
while (count < MaxIts)
    count = count + 1;
    GatingOutput = exp(Input*Weight);
    Grad = (GatingOutput./(SumOtherGatingsOutputs + GatingOutput) - Target);
    Grad = Input'*Grad;
    NewWeight = Weight - 4*InvH*Grad;
    if (norm(NewWeight - Weight)/norm(NewWeight) < MinWeightChange) | (count > MaxIts)
        Weight = NewWeight;
        break;
    end
    Weight = NewWeight;
end