function eval = metric_evaluation(subjective,objective,beta)
warning off

if nargin<3 %values for beta as in VSI implementation
	 beta(1) = 10;
	 beta(2) = 0;
	 beta(3) = mean(objective);
	 beta(4) = 1;
	 beta(5) = 0.1;
end

if(isrow(subjective))
    subjective = subjective';
end

if(isrow(objective))
    objective = objective';
end

try
	spearman= (corr(objective, subjective, 'type', 'spearman'));
	kendall= (corr(objective, subjective, 'type', 'kendall'));

	[bayta, ehat,J] = nlinfit(objective,subjective,@logistic,beta); 
	[ypre, junk] = nlpredci(@logistic,objective,bayta,ehat,J);
	
 
	%RMSE = (sqrt(sum((ypre - subjective).^2) / length(subjective)));
	pearson = (corr(subjective, ypre, 'type','Pearson')) ;
	
catch %just in case
    %disp('case');
	spearman=0;
	kendall=0;    
	pearson=0;    
	RMSE=0;    
end
eval=[pearson, spearman, kendall];