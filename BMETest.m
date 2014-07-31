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

function [MAE, PMAE, Predictions, MPredictions] = BMETest(EInput, GInput, Target, BME)

%% Test
N = size(EInput,1);
Num = BME.Test.NumExperts;
ExpertsMeans = BMEExpertsMeans(EInput,BME);
GatingsOutputs = exp(GInput*BME.Gatings.Weights);
[values, index] = sort(GatingsOutputs,2,'descend');

MAE = zeros(1,Num);
PMAE = zeros(1,Num);
Results1 = inf*ones(N, 1);
Results2 = zeros(size(Target));
Predictions = zeros(N, Num);
if size(Target,2) == 1
    MPredictions = zeros(N,Num);
else
    MPredictions = zeros(N,size(Target,2),Num);
end

for i = 1:Num
    for j = 1:N
        if size(Target,2) == 1
            Results1(j) = min(Results1(j), abs(Target(j) - ExpertsMeans(j,index(j,i))));
            aaa = GatingsOutputs(j,:);
            tindex = index(j, 1:i);
            aaa = aaa/sum(aaa(tindex));
            Results2(j) = ExpertsMeans(j,tindex)*aaa(tindex)';
        else
            Results1(j) = min(Results1(j), mean(abs(Target(j,:) - ExpertsMeans(j,:,index(j,i)))));
            aaa = GatingsOutputs(j,:);
            tindex = index(j, 1:i);
            aaa = aaa/sum(aaa(tindex));
            for t = 1:i
                Results2(j,:) = Results2(j,:) + ExpertsMeans(j,:,tindex(i))*aaa(tindex(i));
            end
        end
    end
    MAE(1,i) = mean(Results1);
    PMAE(1,i) = mean(abs(Results2(:) - Target(:)));
    Predictions(:,i) = Results1;
    if size(Target,2) == 1
        MPredictions(:,i) = Results2;
    else
        MPredictions(:,:,i) = Results2;
    end
end