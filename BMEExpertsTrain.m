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

function BME = BMEExpertsTrain(Target, BME, i)

%% Train the expert of BME
GatingPosterior = BME.Gatings.Posteriors(:,i);
switch lower(BME.Experts.Type)
    case 'rr'     
        EInput = ((GatingPosterior.^0.5)*ones(1,size(BME.Experts.Input,2))) .* BME.Experts.Input ; 
        WeightTarget = repmat(GatingPosterior.^0.5,1,size(Target,2)).*Target ;
        Hessian  = EInput'*EInput + BME.Experts.Alpha*eye(size(EInput,2));
        ExpertWeight = Hessian\(EInput'*WeightTarget);
        if size(Target,2) == 1
            BME.Experts.Weights(:,i) = ExpertWeight;
        else
            BME.Experts.Weights(:,:,i) = ExpertWeight;
        end
        
    case 'rvm'
        if strcmpi(BME.Experts.Kernel, 'linear')
            ED = size(BME.Experts.Input,2);
            EInput = ((GatingPosterior.^0.5)*ones(1,size(BME.Experts.Input,2))) .* BME.Experts.Input; 
            WeightTarget = (GatingPosterior.^0.5).*Target ;
            MaxIt = 200;
            InitAlpha = BME.Experts.Alpha;
            InitBeta = 1./BME.Experts.Variances(i);
            [ExpertWeight, Used, Beta] = RVMRegressor(EInput, WeightTarget, InitAlpha*ones(ED,1), InitBeta, MaxIt);
            BME.Experts.Weights(:,i) = ExpertWeight;
            BME.Experts.Used{i} = Used;
            BME.Experts.Beta(1,i) = Beta;
        else
            Threshold = 1/BME.NumExperts/3;
            Index = find(GatingPosterior > Threshold);
            Index = [1; Index+1];
            EInput = BME.Experts.Input(:,Index);
            EInput = ((GatingPosterior.^0.5)*ones(1,size(EInput,2))) .* EInput; 
            WeightTarget = (GatingPosterior.^0.5).*Target;
            MaxIt = 200;
            InitAlpha = BME.Experts.Alpha;
            InitBeta = 1./BME.Experts.Variances(i);
            [ExpertWeight, Used, Beta] = RVMRegressor(EInput, WeightTarget, InitAlpha*ones(length(Index),1), InitBeta, MaxIt);
            Used = Index(Used);
            BME.Experts.Weights(Index,i) = ExpertWeight;
            BME.Experts.Used{i} = Used;
            BME.Experts.Beta(1,i) = Beta;
        end
        
    case 'frvm'
        if strcmpi(BME.Experts.Kernel, 'linear')
            ED = size(BME.Experts.Input,2);
            Nbf = round(BME.Experts.Nbf*ED);
            EInput = ((GatingPosterior.^0.5)*ones(1,ED)) .* BME.Experts.Input;
            if size(Target,2) == 1
                WeightTarget = (GatingPosterior.^0.5).*Target;
                [ExpertWeight, Used, SAlpha, ICov] = FRVM(EInput, WeightTarget, BME.Experts.Variances(i), Nbf);      
                BME.Experts.Weights(:,i) = zeros(ED,1);
                BME.Experts.Weights(Used,i) = ExpertWeight;
                BME.Experts.Used{i} = Used;
                BME.Experts.SAlpha{i} = SAlpha;
                BME.Experts.ICov{i} = ICov;
            else
                for j = 1:size(Target,2)
                    WeightTarget = (GatingPosterior.^0.5).*Target(:,j);
                    [ExpertWeight, Used, SAlpha, ICov] = FRVM(EInput, WeightTarget, BME.Experts.Variances(j,i), Nbf);      
                    BME.Experts.Weights(:,j,i) = zeros(ED,1);   
                    BME.Experts.Weights(Used,j,i) = ExpertWeight;
                    BME.Experts.Used{j,i} = Used;
                    BME.Experts.SAlpha{j,i} = SAlpha;
                    BME.Experts.ICov{j,i} = ICov;
                end
            end
        else
            Threshold = 1/BME.NumExperts/3;
            ED = size(BME.Experts.Input,2);
            Index = find(GatingPosterior > Threshold);
            Index = [1; Index+1];
            Nbf = round(BME.Experts.Nbf*length(Index));
            EInput = BME.Experts.Input(:,Index);
            EInput = ((GatingPosterior.^0.5)*ones(1,size(EInput,2))) .* EInput;
            if size(Target,2) == 1
                WeightTarget = (GatingPosterior.^0.5).*Target;
                [ExpertWeight, Used, SAlpha, ICov] = FRVM(EInput, WeightTarget, BME.Experts.Variances(i), Nbf);
                Used = Index(Used);
                BME.Experts.Weights(:,i) = zeros(ED,1);
                BME.Experts.Weights(Used,i) = ExpertWeight;
                BME.Experts.Used{i} = Used;
                BME.Experts.SAlpha{i} = SAlpha;
                BME.Experts.ICov{i} = ICov;
            else
                for j = 1:size(Target,2)
                    WeightTarget = (GatingPosterior.^0.5).*Target(:,j);
                    [ExpertWeight, Used, SAlpha, ICov] = FRVM(EInput, WeightTarget, BME.Experts.Variances(j,i), Nbf);
                    Used = Index(Used);
                    BME.Experts.Weights(:,j,i) = zeros(ED,1);
                    BME.Experts.Weights(Used,j,i) = ExpertWeight;
                    BME.Experts.Used{j,i} = Used;
                    BME.Experts.SAlpha{j,i} = SAlpha;
                    BME.Experts.ICov{j,i} = ICov;
                end
            end
        end
        
    otherwise
        disp('Unknown method.')
end