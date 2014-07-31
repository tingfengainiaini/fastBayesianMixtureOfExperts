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

function GatingOutputs = BMEGatingsOutputsNorm(BME)
%% Normalize the output of gating network

GatingsOutputs = BME.Gatings.Outputs;
SumGatingsOutput = sum(GatingsOutputs,2) ;
GatingOutputs = GatingsOutputs./repmat(SumGatingsOutput,1,size(GatingsOutputs,2)) ;