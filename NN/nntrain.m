function [nn, L]  = nntrain(nn, train_x, train_y, opts, val_x, val_y)
%NNTRAIN trains a neural net
% [nn, L] = nnff(nn, x, y, opts) trains the neural network nn with input x and
% output y for opts.numepochs epochs, with minibatches of size
% opts.batchsize. Returns a neural network nn with updated activations,
% errors, weights and biases, (nn.a, nn.e, nn.W, nn.b) and L, the sum
% squared error for each training minibatch.
err=[];
assert(isfloat(train_x), 'train_x must be a float');
assert(nargin == 4 || nargin == 6,'number ofinput arguments must be 4 or 6')

loss.train.e               = [];
loss.train.e_frac          = [];
loss.val.e                 = [];
loss.val.e_frac            = [];
opts.validation = 0;
if nargin == 6
    opts.validation = 1;
end

% fhandle = [];
% if isfield(opts,'plot') && opts.plot == 1
%     fhandle = figure();
% end

m = size(train_x, 1);

batchsize = opts.batchsize;
numepochs = opts.numepochs;

% numbatches = m / batchsize;
numbatches = floor(m / batchsize);

assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');

L = zeros(numepochs*numbatches,1);
n = 1;
TrnMAE=[];%zeros();
trnOut=[];
for i = 1 : numepochs

    
    if i>nn.momEpo
        nn.momentum=nn.finalMomentum;
    end
    
    kk = randperm(m);
    for l = 1 : numbatches
        batch_x = train_x(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        
        %Add noise to input (for use in denoising autoencoder)
        if(nn.inputZeroMaskedFraction ~= 0)
            batch_x = batch_x.*(rand(size(batch_x))>nn.inputZeroMaskedFraction);
        end
        
        batch_y = train_y(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        
        nn = nnff(nn, batch_x, batch_y);
        
        if i == numepochs % error for the last epo to clc thr
            trnOut=[trnOut;nn.e]; 
        end
        nn = nnbp(nn);
        nn = nnapplygrads(nn);
        
        L(n) = nn.L;
        n = n + 1;
        
        
        
    end
    

    
    if opts.validation == 1
        loss = nneval(nn, loss, train_x, train_y, val_x, val_y);
        str_perf = sprintf(';\n Full-batch train mse = %f, val mse = %f',...
            loss.train.e(end), loss.val.e(end));
        err=[err;loss.val.e(end)];
    else
        loss = nneval(nn, loss, train_x, train_y);
        str_perf = sprintf(';\n Full-batch train err = %f', loss.train.e(end));
    end
    %     if ishandle(fhandle)
    %         nnupdatefigures(nn, fhandle, loss, opts, i);
    %     end
    
    %     if  i==numepochs %(i==1 || rem(i/10,1)==0 || i==numepochs)
    %
    %     disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) '. Took '...
    %         num2str(t) ' seconds.'...
    %         ,' Mini-batch mean squared error on training set is '...
    %         num2str(mean(L((n-numbatches):(n-1)))) str_perf]);
    %    end
    nn.learningRate = nn.learningRate * nn.scaling_learningRate;
end

nn.trnOut=trnOut;
% For Autoencoder
if strcmp(nn.alg,'Auto')
    TrnErrors=mean(abs(nn.trnOut),2);
    TrnMAE=[TrnMAE;TrnErrors]; %calc error for Autoencoder
%     train_x_mean=mean(train_x); %mean for all features
%     train_x_cov=cov(train_x(:,1:size(train_x,2)));
%     for i=1:size(train_x,1)
%     dis2_mean(i)=pdist2(train_x(i,1:size(train_x,2)),train_x_mean,'mahalanobis',train_x_cov);
%     end
%     nn.thrs=chi2inv(0.99,size(train_x,2)); %99%
%     rst=dis2_mean<nn.thrs;

    nn.thrs=mean(TrnMAE)+3*std(TrnMAE);
%     nn.thrs=mean(TrnMAE)+std(TrnMAE);
% nn.thrs=mean(TrnMAE)+2*std(TrnMAE);
end

end

