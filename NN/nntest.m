%function [er, bad] = nntest(nn, x, y)
function [result] = nntest(nn, x, y)
    [out,~] = nnpredict(nn, x);
    
    if strcmp(nn.alg, 'DBN')==1
        rst=out>.5; 
    elseif strcmp(nn.alg, 'Auto')==1
        MAEout=mean(abs(x-out),2);
        rst=MAEout<=nn.thrs; %if record deteced correctly return 1
        thrs=nn.thrs;
    else
        error('Input algorithm shoul be Auto or DBN.');      
    end
    
    result=anoResult(y,rst);
end
