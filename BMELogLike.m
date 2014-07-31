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

function LogLike = BMELogLike(Target, BME)
%% Compute the loglikelihood

ExpertsMeans = BME.Experts.Means;
ExpertsVariances = BME.Experts.Variances;
GatingsOutputs = BMEGatingsOutputsNorm(BME);

sumprob = 0.0;
if size(Target,2) == 1
    for i = 1:BME.NumExperts
        sumprob = sumprob + GatingsOutputs(:,i).*onedimgauss(ExpertsMeans(:,i) - Target, ExpertsVariances(i));
    end
else
    for i = 1:BME.NumExperts
        productprob = GatingsOutputs(:,i);
        for j = 1:size(Target,2)
            productprob = productprob.*onedimgauss(ExpertsMeans(:,j,i) - Target(:,j), ExpertsVariances(j,i));
        end
        sumprob = sumprob + productprob;
    end
end
LogLike = sum(log(sumprob));