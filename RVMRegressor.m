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

function [weights, used, beta, marginal, alpha, gamma] = RVMRegressor(EInput, Target, alpha, beta, MaxIt)
%% Estimate hyperparameters for a sparse Bayesian model

%% Terminate estimation when no log-alpha value changes by more than this
MIN_DELTA_LOGALPHA	= 1e-3;
% Prune basis function when its alpha is greater than this
ALPHA_MAX		= 1e6 ; 
% Iteration number during training where we switch to 'analytic pruning'
PRUNE_POINT		= 50;		% percent

%% Also Set Alpha_min 
ALPHA_MIN = 1e-20 ; 

[N,M]	= size(EInput);
w	= zeros(M,1);
EInputTarget	= EInput'*Target;
Hessian = EInput'*EInput;
%%%%%%%%%%%%%%%%%%%%%%%%%%%

gamma		= ones(M,1);
nonZero		= logical(ones(M,1));
% PRUNE_POINT	= MaxIt * (PRUNE_POINT/100);
LAST_IT		= 0;

% disp(['Maximum Iterations :' num2str(MaxIt)]); 
for i=1:MaxIt
    
  pNonZero = nonZero ;  
  nonZero = (alpha < ALPHA_MAX) ;
  
%   fprintf('i = %d\n',i);
%   fprintf('nonzero = %d\n',sum(nonZero));
  
  if(~sum(nonZero))
      nonZero = pNonZero ;
      alpha = pAlpha ; 
      break ; 
  end
  alpha_nz	= alpha(nonZero);
  
  w(~nonZero)	= 0;
  M		= sum(nonZero);  
  % Work with non-pruned basis
  % 
  TempHessian	= Hessian(nonZero, nonZero);
  TempHessian   = TempHessian + 1e-7*min(diag(TempHessian))*eye(M);
  TempHessian	= TempHessian*beta + diag(alpha_nz) ;

  %% Add a loop here to make Hessian Positive definite

  [U,  p]= chol(TempHessian);

  lambda = 1e-5*min(diag(TempHessian));
  while (p > 0)
      TempHessian = TempHessian + lambda*eye(numAlpha_nz) ;
      lambda = lambda*10 ;
      [U, p] = chol(TempHessian) ;
  end 

  Ui		= inv(U);
  % Quick ways to get diagonal of posterior weight covariance matrix 'SIGMA'
  diagSig	= sum(Ui.^2,2);
  % well-determinedness parameters
  gamma		= 1 - alpha_nz.*diagSig;
%   if sum(gamma) > 250
%       aaa=1;
%   end
 
  %% Add a loop here to make gamma positive
  while (min(gamma) < 0)
      TempHessian = TempHessian + lambda*eye(numAlpha_nz);
      lambda = lambda*10;
      [U, p] = chol(TempHessian);
      Ui = inv(U);
      diagSig = sum(Ui.^2,2);
      gamma		= 1 - alpha_nz.*diagSig;
  end
  
  w(nonZero)	= (Ui * (Ui' * EInputTarget(nonZero)))*beta;
  ED		= sum((Target - EInput*w).^2); % Data error
  betaED	= beta*ED;
  logBeta	= N*log(beta);
  
  % Quick ways to get determinant
  logdetH	= -2*sum(log(diag(Ui)));

  % Compute marginal likelihood (approximation for classification case)
  marginal	= -0.5* (logdetH - sum(log(alpha_nz)) - ...
			 logBeta + betaED + (w(nonZero).^2)'*alpha_nz);

  % Output info if requested and appropriate monitoring iteration
%   if ((verbose == 1) && LAST_IT)    
%      fprintf('%5d> L = %.3f\t Gamma = %.2f (nz = %d)\t s=%.3f\n',...
%             i, marginal, sum(gamma), sum(nonZero), sqrt(1/beta));    
%   end

  if ~LAST_IT
    % alpha and beta re-estimation on all but last iteration
    % (we just update the posterior statistics the last time around)
   logAlpha		= log(alpha(nonZero)) ;
   pAlpha  = alpha ; 
   
   %if i<PRUNE_POINT
      % MacKay-style update given in original NIPS paper
      alpha(nonZero)	= gamma ./ w(nonZero).^2;
    
      %else
      % Hybrid update based on NIPS theory paper and AISTATS submission
      %alpha(nonZero)	= gamma ./(w(nonZero).^2./gamma - diagSig);
      %alpha(alpha<=0)	= inf ;   % This will be pruned later
   
  %end
    
   anz		= alpha(nonZero);
   maxDAlpha	= max(abs(logAlpha(anz~=0)-log(anz(anz~=0))));
    
   % Terminate if the largest alpha change is judged too small
   if maxDAlpha < MIN_DELTA_LOGALPHA
     LAST_IT	= 1;
%      if verbose == 1
%         fprintf('Terminating: max log(alpha) change is %g (<%g).\n', ...
%                  maxDAlpha, MIN_DELTA_LOGALPHA);
%       end
    end
        
    % Beta re-estimate in regression (unless fixed) 
    beta  = abs(N - sum(gamma))/ED  ;
   
  else
      
    % Its the last iteration due to termination, leave outer loop
    break;	% that's all folks!
  end
  
end

% Tidy up return values
weights	= w;
used	= find(nonZero);
  
 