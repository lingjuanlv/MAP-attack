% function [er, bad] = nntest(nn, x, y)
function [result] = nntest_llv(nn, x, y)
    [out,~] = nnpredict(nn, x); % predict out
    
    if strcmp(nn.alg, 'DBN')==1
        rst=out>.5; 
    elseif strcmp(nn.alg, 'Auto')==1
        MAEout=mean(abs(x-out),2);  % mean(X,1),colum;mean(X,2),row
        rst=MAEout<=nn.thrs; % MAEout<threshold 1 means normal,otherwise 0 anomaly
        rst=~rst; %for labeling,we label normal as 0,anomaly as 1
        thrs=nn.thrs;

%%
% use mahalanobis to mean vector to get predicted label
%     train_x_mean=mean(x); %mean for all features
%     train_x_cov=cov(x(:,1:size(x,2)));
%     for i=1:size(x,1)
%     dis2_mean(i)=pdist2(x(i,1:size(x,2)),train_x_mean,'mahalanobis',train_x_cov);
%     end
%     nn.thrs=chi2inv(0.99,size(x,2)); %99%
%     rst=dis2_mean'<nn.thrs;
    
    else
        error('Input algorithm shoulbe Auto or DBN.');      
    end
    
    result=anoResult_llv(y,rst);
end
