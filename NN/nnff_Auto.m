function nn = nnff_Auto(nn, x)
% Initialize some variables
n = nn.n; 
mid_n=ceil(n / 2);
m = size(x, 1);

nn.a{1} = [ones(m,1) x];

for i = 2 : n-1
     if i ~= mid_n %&& i ~= n
            % Calculate the unit's outputs (including the bias term)
            nn.a{i} = sigm(nn.a{i - 1} * nn.W{i - 1}');
     else
           nn.a{i} = nn.a{i - 1} * nn.W{i - 1}';
    end
    
    %dropout
    if(nn.dropoutFraction > 0)
        if(nn.testing)
            nn.a{i} = nn.a{i}.*(1 - nn.dropoutFraction);
        else
            nn.dropOutMask{i} = (rand(size(nn.a{i}))>nn.dropoutFraction);
            nn.a{i} = nn.a{i}.*nn.dropOutMask{i};
        end
    end
    
    %calculate running exponential activations for use with sparsity
    if(nn.nonSparsityPenalty>0)
        nn.p{i} = 0.99 * nn.p{i} + 0.01 * mean(nn.a{i}, 1);
    end
    
    %Add the bias term
    if i ~= n
         nn.a{i} = [ones(m,1) nn.a{i}];
    end
end
%output
nn.a{n} = sigm(nn.a{n - 1} * nn.W{n - 1}');

%error and loss
nn.e = x - nn.a{n};

nn.L = 1/(2*m) * sum(sum(nn.e .^ 2));
