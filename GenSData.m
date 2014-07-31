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

clear
t = rand(200,1);
t = [t; (0:0.005:0.999)'];
X(1:200,1) = t(1:200) + 0.3*sin(2*pi*t(1:200)) + 0.05*randn(size(t(1:200)));
X(201:400,1) = t(201:400) + 0.3*sin(2*pi*t(201:400));

[Input, index] = sort(X(1:200,1));
TestInput = X(201:end,1);
Target = t(1:200,1);
Target = Target(index);
TestTarget = t(201:end,1);

save SData Input TestInput Target TestTarget 