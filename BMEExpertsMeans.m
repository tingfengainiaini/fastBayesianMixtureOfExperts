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

function ExpertsMeans = BMEExpertsMeans(Input,BME)
%% Compute the mean of experts

if length(size(BME.Experts.Weights)) == 2
    ExpertsMeans = Input*BME.Experts.Weights;
else
    for i = 1:size(BME.Experts.Weights,3)
        ExpertsMeans(:,:,i) = Input*BME.Experts.Weights(:,:,i);
    end
end