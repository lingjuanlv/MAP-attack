
function [out,labels] = nnpredict(nn, x)
    nn.testing = 1;
    nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));
    nn.testing = 0;
    out=nn.a{end}; %output
     [~, i] = max(nn.a{end},[],2);
     labels = i;
end
