% function test_example_SAE
clear all; close all; clc;

addpath '../data'
addpath '../util'
addpath '../NN'
addpath '../SAE'

load('preprocess_datasets/Gaussian-n=15,sigma=1.0.mat')
data=X';
NoAno=round(size(data,2)/20);
NoTrn=round(2*size(data,2)/3);

% anomaly=rand(size(D,1), NoAno);
% inputs=([D,anomaly]);

% load OppSubset.mat 
%load BananaDS.mat

d_x = (data(:,1:800))';      %800 train
anomaly_tr=genAnomaly(d_x);
NoAno_x=size(anomaly_tr(1:40,:),1);
train_x=[d_x;anomaly_tr(1:NoAno_x,:)];


d_v=(data(:,801:1000))';    %200 validation
anomaly_v=genAnomaly(d_v);
NoAno_v=size(anomaly_v(1:10,:),1);
val_x=[d_v;anomaly_v(1:NoAno_v,:)];

d_t  = (data(:,1001:1200))'; %200 test
anomaly=anomaly_tr(1:10,:);
NoAno=size(anomaly,1);
test_x=[d_t;anomaly];


train_y = [zeros(size(d_x,1),1);...
    ones(NoAno_x,1)];
val_y   = [zeros(size(d_v,1),1);...
    ones(NoAno_v,1)];
test_y  = [zeros(size(d_t,1),1);...
    ones(NoAno,1)];

output=1;
h1=100;
h2=100;

%%  ex1 train a 100 hidden unit SDAE and use it to initialize a FFNN
%  Setup and train a stacked denoising autoencoder (SDAE)

% rand('state',0)
% sae = saesetup([784 100]);
% sae.ae{1}.activation_function       = 'sigm';
% sae.ae{1}.learningRate              = 1;
% sae.ae{1}.inputZeroMaskedFraction   = 0.5;
% opts.numepochs =   1;
% opts.batchsize = 100;
% sae = saetrain(sae, train_x, opts);
% visualize(sae.ae{1}.W{1}(:,2:end)')

% %Use the SDAE to initialize a FFNN
% nn = nnsetup([784 100 10]);
% nn.activation_function              = 'sigm';
% nn.learningRate                     = 1;
% nn.W{1} = sae.ae{1}.W{1};
% 
% % Train the FFNN
% opts.numepochs =   1;
% opts.batchsize = 100;
% nn = nntrain(nn, train_x, train_y, opts);
% [er, bad] = nntest(nn, test_x, test_y);
% assert(er < 0.16, 'Too big error');

%% ex2 train a 100-100 hidden unit SDAE and use it to initialize a FFNN
%  Setup and train a stacked denoising autoencoder (SDAE)
rand('state',0)
sae = saesetup([size(train_x,2) h1 h2]);
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 1;
sae.ae{1}.inputZeroMaskedFraction   = 0.5;

sae.ae{2}.activation_function       = 'sigm';
sae.ae{2}.learningRate              = 1;
sae.ae{2}.inputZeroMaskedFraction   = 0.5;

opts.numepochs =   1;
opts.batchsize = 105;
sae = saetrain(sae, train_x, opts);
%visualize(sae.ae{1}.W{1}(:,2:end)')

% Use the SDAE to initialize a FFNN
nn = nnsetup([size(train_x,2) h1 h2 output]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 1;

%add pretrained weights
nn.W{1} = sae.ae{1}.W{1};
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   3;
opts.batchsize = 105;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y)
assert(er < 0.1, 'Too big error');
