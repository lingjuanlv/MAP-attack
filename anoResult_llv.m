function result=anoResult_llv(y,predicted_L)  %y:true label,predicted_L:predicted label accoding to threshold

mis=find(y~=predicted_L); %Find indices of y~=pr_y.

FN=sum(predicted_L==0&y==1);
TP=sum(predicted_L==1&y==1);
FNR=FN/(FN+TP);    %FNR=FN/(TP+FN)


FP=sum(predicted_L==1&y==0);
TN=sum(predicted_L==0&y==0);
FPR=FP/(FP+TN);    %FPR=FP/(TN+FP)

ACC = (TP + TN) / (FN + TP+FP+TN); %accu = (TP+TN)/(TP+FN+FP+TN)
AUC= (length(y)- length(mis)) /length(y);
result=[FP,FPR,FN,FNR,AUC];
end