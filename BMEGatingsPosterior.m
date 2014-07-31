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

function GatingsPosterior = BMEGatingsPosterior(Target, BME)

%% Compute the posterior probability. 

GatingsOutputs = BMEGatingsOutputsNorm(BME);
Means = BME.Experts.Means;
Variances = BME.Experts.Variances;
GatingsPosterior = zeros(size(GatingsOutputs));

if size(Target,2) == 1
    for i = 1:BME.NumExperts
        GatingsPosterior(:,i) = GatingsOutputs(:,i).*onedimgauss(Means(:,i) - Target, Variances(i));
    end
else
    for i = 1:BME.NumExperts
        for j = 1:size(Target,2)
            GatingsPosterior(:,i) = GatingsOutputs(:,i).*onedimgauss(Means(:,j,i) - Target(:,j), Variances(j,i)) ;
        end
    end
end
SumGatingsPosterior = sum(GatingsPosterior,2) ;
GatingsPosterior = GatingsPosterior./repmat(SumGatingsPosterior,1,size(GatingsPosterior,2)) ;
for i = 1:BME.NumExperts
    index = find(GatingsPosterior(:,i) < 1e-10);
    GatingsPosterior(index,i) = 0;
end