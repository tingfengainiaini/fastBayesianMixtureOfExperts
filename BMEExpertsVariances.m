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

function Variances = BMEExpertsVariances(Target, BME)

%% Compute the variance of expert

N = length(Target);
Num = BME.NumExperts;
Threshold = 0.1/Num;

ExpertMeans = BME.Experts.Means;
GatingsPosteriors = zeros(N,Num);
for i = 1:Num
    index = find(BME.Gatings.Posteriors(:,i) > Threshold);
    GatingsPosteriors(index,i) = BME.Gatings.Posteriors(index,i);
end

sumQ = sum(GatingsPosteriors) ; 
d2 = size(Target,2);
if d2 == 1
    numV = sum(GatingsPosteriors.*((repmat(Target,1,size(ExpertMeans,2)) - ExpertMeans).^2)) ; 
else
    for i = 1:Num
        numV(:,i) = sum(repmat(GatingsPosteriors(:,i),1,d2).*(Target - ExpertMeans(:,:,i)).^2)';
    end;
end
Variances = (numV./(repmat(sumQ,d2,1) + eps));