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

function BME = BMEGatingsTrain(BME, i)
%% Train the gating network of BME

%%
MinWeightChange = 1e-4;
MaxIt = 50;
GatingPosterior = BME.Gatings.Posteriors(:,i);
GatingWeight = BME.Gatings.Weights(:,i);
SumOtherGatingsOutputs = sum(BME.Gatings.Outputs,2) - BME.Gatings.Outputs(:,i);

switch lower(BME.Gatings.Type)  
    case 'mlr'
        % compute the most probable weights by bound optimization
        count = 0;
        while (count < MaxIt)
            count = count + 1;
            aaa = BME.Gatings.Input*GatingWeight;
            GatingOutput = exp(aaa);
%             obj(count,1) = sum(GatingPosterior.*aaa - log(GatingOutput + SumOtherGatingsOutputs))...
%                 - BME.Gatings.Alpha/8*GatingWeight'*GatingWeight;
            Grad = (GatingOutput./(SumOtherGatingsOutputs + GatingOutput) - GatingPosterior);
            Grad = BME.Gatings.Input'*Grad + BME.Gatings.Alpha/4*GatingWeight;
            NewGatingWeight = GatingWeight - 4*BME.Gatings.InvH*Grad;
            if (norm(NewGatingWeight - GatingWeight)/norm(NewGatingWeight) < MinWeightChange) | (count > MaxIt)
                GatingWeight = NewGatingWeight;
                break;
            end
            GatingWeight = NewGatingWeight;
        end
        BME.Gatings.Weights(:,i) = GatingWeight;
    
%          [GatingWeight,run] = BFGSGatings(BME.Gatings.Input,GatingPosterior,BME.Gatings.Weights(:,i),BME.Gatings.Alpha/4,SumOtherGatingsOutputs);
%             BME.Gatings.Weights(:,i) = GatingWeight;
    
    case 'smlr'
        if strcmpi(BME.Gatings.Kernel, 'linear')
            GD = size(BME.Gatings.Input,2);
            Nbf = round(BME.Gatings.Nbf*GD);
            [GatingWeight, Used] = FSLR(BME.Gatings.Input, GatingPosterior, SumOtherGatingsOutputs, Nbf);
            BME.Gatings.Weights(:,i) = zeros(GD,1);
            BME.Gatings.Weights(Used,i) = GatingWeight;
            BME.Gatings.Used{i} = Used;
        else
            Threshold = 1/BME.NumExperts/3;
            GD = size(BME.Gatings.Input,2);
            Index = find(GatingPosterior > Threshold);
            Index = [1; Index+1];
            Nbf = round(BME.Gatings.Nbf*length(Index));
            [GatingWeight, Used] = FSLR(BME.Gatings.Input(:,Index), GatingPosterior, SumOtherGatingsOutputs, Nbf);
            Used = Index(Used);
            BME.Gatings.Weights(:,i) = zeros(GD,1);
            BME.Gatings.Weights(Used,i) = GatingWeight;
            BME.Gatings.Used{i} = Used;
        end
    otherwise
        disp('Unknown method.');
end
