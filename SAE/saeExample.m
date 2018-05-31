clear all; close all; %clc;

addpath '../data'
addpath '../util'
addpath '../NN'
addpath '../../../Daset'
addpath '../../../Daset/Original'
addpath '../../../Utils'

addpath('../implementation')
addpath('../../Utils')
finRslt=[];

%load BananaDS.mat
%data=rand(7500,200);

%load HAR.mat
%load GasSen.mat
load GME25.mat
% %load OpprtDS.mat
% load Opp2.mat
% d1=data;
% load OppTst2.mat
% d2=data;
%  load OppAll.mat
%  data=data(1:35000,1:30);

% % % % SDA-classification
% h=3;%round(size(data,2)/2);
% prcTrn=80;
% prcTst=100-prcTrn;
% prcAno=6;
% perLabel=100;
% batchSize=100;
% 
% prepData;
% sae = saesetup([size(train_x,2) h]);%1 h2
% sae.ae{1}.activation_function       = 'sigm';
% sae.ae{1}.learningRate              = .08;
% sae.ae{1}.inputZeroMaskedFraction   = .5;
% sae.ae{1}.finalMomentum             = .9;
% opts.numepochs =   200;
% opts.batchsize = 100;
% opts.alg='s';
% sae.ae{1}.alg='s';
% sae = saetrain(sae, train_x, opts);
% %visualize(sae.ae{1}.W{1}(:,2:end)')
% 
% % Use the SDAE to initialize a FFNN
% nn = nnsetup([size(train_x,2) h 1]);
% nn.activation_function              = 'sigm';
% nn.learningRate                     = .08;
% nn.finalMomentum                    = .9;
% %nn.momEpo
% nn.alg='DBN';
% nn.W{1} = sae.ae{1}.W{1};
% 
% % Train the FFNN
% opts.numepochs =   100;
% opts.batchsize = 100;
% nn = nntrain(nn, train_x, train_y, opts, val_xn, val_yn);
% out= nntest(nn, test_x, test_y);

% % 


h1=12;%round(size(data,2)/2);
h2=round(h1/2);
prcTrn=20;%80;
prcTst=5;%100-prcTrn;
prcAno=6;
perLabel=100;
batchSize=100;

prepData;
input=size(train_x,2);
sae = saesetup([input h1 h2]);
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = .1;
%sae.ae{1}.inputZeroMaskedFraction   = .5;
sae.ae{1}.finalMomentum             = .9;
sae.ae{1}.weightPenaltyL2           = .000002;

sae.ae{2}.activation_function       = 'sigm';
sae.ae{2}.learningRate              = .1;
%sae.ae{1}.inputZeroMaskedFraction   = .5;
sae.ae{2}.finalMomentum             = .9;
sae.ae{2}.weightPenaltyL2           = .000002;

opts.numepochs =   20;
opts.batchsize = 100;

sae.ae{1}.alg='Auto';
sae.ae{2}.alg='Auto';
sae = saetrain(sae, train_x, opts);


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
 


