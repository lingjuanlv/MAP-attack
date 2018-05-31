%%the same result as TTPRandKey2
function Z=two_stage(inputs,w) %inputs:attributes*records,n*m
n= size(inputs,1); % 1st dimension:attributes,already transposed,like iris data
m=size(inputs,2);   %records
% w=10;   % w<<n
T=rand(w,n);
Y=two_Gompertz(inputs);    %Y:n*m
Z=T*Y;   %Z:w*m matrix,w reduced attributes,m records
end