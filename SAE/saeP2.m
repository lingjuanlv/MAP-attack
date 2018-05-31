% Use the SDAE to initialize a FFNN
nn = nnsetup([input h1 h2 h1 input]);
nn.activation_function              = 'sigm';
nn.learningRate                     = .01;
nn.finalMomentum                    = .8;
%nn.weightPenaltyL2                  = .00002;
%nn.momEpo                           =5;
nn.alg='Auto';
nn.W{1} = sae.ae{1}.W{1};
nn.W{2} = sae.ae{2}.W{1};
nn.W{3} = sae.ae{2}.W{2};
nn.W{4} = sae.ae{1}.W{2};
% Train the FFNN
opts.numepochs =   100;
opts.batchsize = 100;
nn = nntrain(nn, train_x, train_x, opts, val_xn, val_xn);
out= nntest(nn, test_x, test_y);