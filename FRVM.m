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

function [weight, sv, salpha, R] = FRVM(Input, Target, lambda, nbf)

%% Forward relevant vector machine

if (nargin ~= 4) % check correct number of arguments
    
    help FGSSML
    
else

%     fprintf('Full Matching Pursuit ..................\n')
    
    % initialization
    [n, d] = size(Input);
    ddd = sum(Input.^2)';
    R = [];
    sv = [];
    SvInput = [];
    nonbf = 1:d;
    Y = zeros(d, nbf);
    WEIGT = zeros(d, nbf);

%     first iteration
    qq = Input'*Target/lambda;
    ss = ddd/lambda;
    alpha = ss.^2./(qq.^2 - ss);
    pindex = find(alpha > 0);
    score = zeros(length(qq), 1);
    score(pindex) = log(alpha(pindex)) - log(alpha(pindex) + ss(pindex)) + qq(pindex).^2./(alpha(pindex) + ss(pindex));
    
    [temp,tindex] = max(score);
    salpha(1,1) = alpha(tindex);
    index = nonbf(tindex);
    nonbf(tindex) = [];
    
    x = Input(:,index);
    R = 1/(x'*x + 1e-8 + lambda); % compute the inverse matrix
    sv = [sv,index];
    
    % compute the change of posterior
    y = x'*Target;
    weight = R*y;
    SvInput = [SvInput x];
    qq = Input(:,nonbf)'*(Target - SvInput*weight)/lambda;
    
    % compute the change of determinant
    Y(nonbf,1) = x'*Input(:,nonbf);
    WEIGT(nonbf,1) = R*Y(nonbf,1);
    ss = (ddd(nonbf) - sum(Y(nonbf,1).*WEIGT(nonbf,1),2))/lambda;
    
    % print the reuslt
%     Error = mean((Target - Input(:,sv)*weight).^2);
%     disp(['Error: ' num2str(Error)]);
    
    for i = 2:nbf
        
        % select the basis function
        alpha = ss.^2./(qq.^2 - ss);
        pindex = find(alpha > 0);
        if length(pindex) < 1
            break;
        end
        score = zeros(length(qq), 1);
        score(pindex) = log(alpha(pindex)) - log(alpha(pindex) + ss(pindex)) + qq(pindex).^2./(alpha(pindex) + ss(pindex));       
        
        [temp,tindex] = max(score);
        salpha(i,1) = alpha(tindex);
        index = nonbf(tindex);
        nonbf(tindex) = [];
    
        % update the inverse matrix using Woodbury inversion identity
        x = Input(:,index);
        h = SvInput'*x;
        beta = R*h;
        gamma = (x'*x + 1e-8 + lambda - h'*beta);
        R = [R zeros(i-1,1); zeros(1,i-1) 0] + [beta; -1]*[beta' -1]./gamma;
        sv = [sv,index];
        
        % compute the change of posterior
        y = [y; x'*Target];
        weight = [weight; 0] + [beta; -1]*([beta' -1]*y)/gamma;
        SvInput = [SvInput x];
        diff = Target - SvInput*weight;
        qq = Input'*diff/lambda;
        qq = qq(nonbf);
        
        % compute the change of determinant
        Y(:,i) = x'*Input;
        TTT = sum(repmat([beta' -1],length(nonbf),1).*Y(nonbf,1:i),2);
        WEIGT(nonbf,1:i) = WEIGT(nonbf,1:i) + (TTT*[beta' -1])/gamma;
        ss = (ddd(nonbf) - sum(Y(nonbf,1:i).*WEIGT(nonbf,1:i),2))/lambda;
        
        % print the result
%         Error = mean(diff.^2);
%         disp(['Error: ' num2str(Error)]);
    end
end