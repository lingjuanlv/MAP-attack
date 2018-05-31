function [result,result_test]=DAE(X)
%% inputs are all matrix,this version is records*attribute
load('preprocess_datasets/Gaussian-n=15,sigma=1.0.mat')

m=3000;                      % number of records for experiments
D=X(randperm(size(X,1),m), :); %X:records*attribute

NoTrn=round(2*size(D,1)/3);
NoVal=round(1*size(D,1)/6);
NoTest=round(1*size(D,1)/6);

d_x=D(1:NoTrn,:); %records*attribute
NoAno_tr=round(NoTrn/20);
% anomaly_tr=rand(NoAno_tr,size(d_x,2));
train_x=mix_anomaly(d_x,NoAno_tr,size(d_x,2),NoAno_tr); %generate anomaly with extremal values uniformally distributed between 0 and delta or between (1-delta) and 1
% train_x=[d_x;anomaly_tr(1:NoAno_tr,:)]; %train_x:records*attribute

d_v=D(NoTrn+1:NoTrn+NoVal,:);
NoAno_v=round(NoVal/20);
% anomaly_v=rand(NoAno_v,size(d_v,2));
% val_x=[d_v;anomaly_v(1:NoAno_v,:)];
val_x=mix_anomaly(d_v,NoAno_v,size(d_v,2),NoAno_v);

d_t=D(NoTrn+NoVal+1:end,:);
NoAno_t=round(NoTest/20);
% anomaly_t=rand(NoAno_t,size(d_t,2));
% test_x=[d_t;anomaly_t(1:NoAno_t,:)];
test_x=mix_anomaly(d_t,NoAno_t,size(d_t,2),NoAno_t);


train_y = [zeros(size(d_x,1),1);...
    ones(NoAno_tr,1)];     %1st column:label,normal 0,anomaly 1
val_y   = [zeros(size(d_v,1),1);...
    ones(NoAno_v,1)];
test_y  = [zeros(size(d_t,1),1);...
    ones(NoAno_t,1)];

%%

% train_x= % give train data
% val_x = %give validation data
% train_y= % train label
% test_x= % test data
% test_y= % test lable

opts.numepochs                      =  10;
opts.batchsize                      = 100;
hidLayer1                           = floor(size(train_x,2)/2);
hidLayer2                           = floor(size(train_x,2)/4);
sae.learningRate                    = .01;
sae.weightPenaltyL2                 = .0;% 1;
sae.nonSparsityPenalty              = .0;
sae.jacobianPenalty                 = .0;

sae = saesetup(sae,[size(train_x,2) hidLayer1 hidLayer2]);
sae = saetrain(sae, train_x, opts);

% Use the SDAE to initialize a FFNN
nn = nnsetup([size(train_x,2) hidLayer1 hidLayer2 hidLayer1 size(train_x,2)]);
epo=opts.numepochs;
opts.alpha=sae.ae{1}.learningRate;
nn.learningRate                     = .1;
nn.finalMomentum                    = .9;
nn.alg                              ='Auto';
nn.jacobianPenalty                 = sae.jacobianPenalty;
saeUnfold;

% Train the FFNN
opts.numepochs                      =   50;
opts.batchsize                      = 100;
nn = nntrain(nn, train_x, train_x, opts, val_x, val_x);

% test 
result= nntest_llv(nn,  train_x, train_y);
result_test= nntest_llv(nn, test_x, test_y);
end