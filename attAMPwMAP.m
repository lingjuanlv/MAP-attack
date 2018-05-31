rng(1);

%- - - - - - - -Configurable- - - - - - - -
dataset=1;     		% 0=multivariate Gaussian
               		% 1=Abalone
               		% 2=Banana
scheme=8;      		% 0=random projection
               		% 1=random transformation
               		% 2=tanh + random transformation
               		% 3=double logistic + random transformation
               		% 4=repeated double logistic + random transformation
               		% 5=staircase + random transformation
               		% 6=mixed tanh and repeated double logistic + random transformation
               		% 7=4th-order polynomial
               		% 8=7th-order polynomial
nPtcps=100;		 		% number of participants
nRowsPP=20;    		% number of data records per participant
m=nPtcps*nRowsPP;	% number of data records
nRPtcps=0.2*nPtcps;
nLeaked=nRPtcps*nRowsPP;
nRecovd=nRowsPP;	% in full simulations, set this to (m-nLeaked)
nOutliers=nRecovd;
fRecovOut=0;   		% 0=recovers normal points; 1=recover outliers
epsilon=0.1;   		% relative error threshold for calc recovery rate
nRuns=1;       		% simulation runs

%- - - - - - - -Early declarations and definitions- - - - - - - -
global pdfT;
global pdfY;

%- - - - - - - -Generate/load data- - - - - - - -
if dataset==0
  n=10;      % number of attributes
  w=5;       % number of attributes upon projection
  sigmaA=1;  % for generating covariance matrix of original data

  mux=zeros(1,n);
  for i=1:n
    mux(i)=i-1;
  end
  sigmafactor=sigmaA*randn(n,n);
  sigmax=(1/n)*(sigmafactor*sigmafactor');
  X=Normalise((mvnrnd(mux,sigmax,m))');  
elseif dataset==1
  ds=load('abalone_dataset');
  if size(ds.abaloneInputs,2)<m
    error('Problem with Abalone dataset');
  end
  n=size(ds.abaloneInputs,1);
  w=n-1;
  X=Normalise(ds.abaloneInputs(:,1:m));
elseif dataset==2
  ds=load('../Dataset/Opportunity.mat');
  if size(ds.data,1)<m 
    error('Problem with Opportunity dataset');
  end
  n=size(ds.data,2);
  w=n-1;
	X=ds.data(1:m,:)';
elseif dataset==3
  ds=load('../Dataset/GAS.mat');
  if size(ds.data,1)<m
    error('Problem with Gas dataset');
  end
  n=size(ds.data,2);
  w=n-1;
	X=ds.data(1:m,:)';
elseif dataset==4
  ds=load('../Dataset/DSA.mat');  
  if size(ds.data,1)<m || size(ds.data,2)<7
    error('Problem with DSA dataset');
  end
  n=size(ds.data,2);
  w=n-1;
	X=ds.data(1:m,:)';
elseif dataset==5
  ds=load('../Dataset/HAR.mat');  
  if size(ds.data,1)<m
    error('Problem with HAR dataset');
  end
  n=size(ds.data,2);
  w=n-1;
	X=ds.data(1:m,:)';
end

mean_recovrate=[0, 0];  % element 0 defined by sang12effective
                        % element 1 defined by me
for sim=1:nRuns
  % permute columns
  X=X(:,randperm(m));
  
  if fRecovOut
    % generate and append outliers
    X=[X,genExtremal(n,nOutliers)];
    % shuffle outliers to just after leaked columns
    X=[X(:,1:nLeaked),X(:,(m+1):(m+nOutliers)),X(:,(nLeaked+1):m)];
  end
  
  %- - - - - - - -Perturb data- - - - - - - -
	if scheme==0 % random perturbation
		Y=X;
	elseif scheme==1 % random transformation
		Y=X;
	elseif scheme==2 % tanh + random transformation
		beta=1.23;
		Y=tanh(beta*X);
	elseif scheme==3 % double logistic + random transformation
		beta=2.81;
		Y=sign(X).*(1-exp(-beta*X.^2));
	elseif scheme==4 % repeated double logistic + random transformation
		beta=16.87;
		Y=sign(X-0.5).*(0.5-0.5*exp(-beta*(X-0.5).^2))+0.5; 
	elseif scheme==5 % staircase + random transformation
		Y=staircase(X);
	elseif scheme==6 % mixed tanh and repeated double logistic + random transformation
		Y=mixedTanhAndRdl(X);
	elseif scheme==7 % 4th-order polynomial
		Y=poly4(X);
	elseif scheme==8 % 7th-order polynomial
		Y=poly7(X);
	end

	T=cell(1,nPtcps);
	for ptcp=1:nPtcps
		colrange=((ptcp-1)*nRowsPP+1):((ptcp-1)*nRowsPP+nRowsPP);
		if scheme==0
			T{ptcp}=2*randn(w,n); % sang12effective: s.d. of T's elements has no impact
		else
			T{ptcp}=rand(w,n);
		end
		Z(:,colrange)=T{ptcp}*Y(:,colrange);
	end
  
  % subplot(2,2,1); plot(X(1,:),X(2,:),'+');
  % subplot(2,2,2); plot(X(1,:),X(3,:),'+');
  % subplot(2,2,3); plot(X(2,:),X(3,:),'+');
  % subplot(2,2,4); plot(Z(1,:),Z(2,:),'+');

  %- - - - - - - -Attack- - - - - - - -
  % init data structures
  Xhat=zeros(size(X,1),nRecovd);
  Yhat=zeros(size(Y,1),nRecovd); 

	% estimate T's pdf based on leaked T's
	% TODO
	pdfT=kde(...,'lcv',ones(1,nLeaked),'Epan');

	% estimate data pdf based on leaked data
	pdfY=kde(Y(:,1:nLeaked),'lcv',ones(1,nLeaked),'Epan');

	% the first nLeaked records are leaked
	% the next nRecovd records are to be recovered
	for j=1:nRecovd
		fprintf('%d..',j);
		z=Z(:,nLeaked+j);
		
		opts=optimset('Algorithm','interior-point');
		problem=createOptimProblem('fmincon','objective',@attAMPwMAP_obf,...
			'nonlcon',attAMPwMAP_con,'lb',zeros(n,1),'ub',ones(n,1),'options',opts);
		gs=GlobalSearch; % MultiStart is not better
		Yhat(:,j)=run(gs,problem);
	end

  if scheme==0 || scheme==1 % random perturbation, random transformation
    Xhat=Yhat;
  elseif scheme==2 % tanh + random transformation
    Xhat=atanh(Yhat)/beta;
  elseif scheme==3 % double logistic + random transformation
    Xhat=sign(Yhat).*((1/beta)*log(1./(1-abs(Yhat)))).^(1/2);
  elseif scheme==4 % repeated double logistic + random transformation
    for i=1:n
      for j=1:nRecovd
        [Xhat(i,j),fval,exitflag,output]...
          =fzero(@(x)sign(x-0.5)*(0.5-0.5*exp(-beta*(x-0.5)^2))+0.5-Yhat(i,j),Yhat(i,j),optimset('display','off'));
        fprintf('  (%f,%f),Yhat=%f,exitflag=%d\n',...
          Xhat(i,j),fval,Yhat(i,j),exitflag);
        if exitflag~=1
          Xhat(i,j)=Yhat(i,j);
        end
      end
    end
  elseif scheme==5 % staircase + random transformation
    Xhat=istaircase(Yhat);
  elseif scheme==6
    for i=1:n
      for j=1:nRecovd
        [Xhat(i,j),fval,exitflag,output]...
          =fzero(@(x)mixedTanhAndRdl(x)-Yhat(i,j),Yhat(i,j),optimset('display','off'));
        fprintf('  (%f,%f),Yhat=%f,exitflag=%d\n',...
          Xhat(i,j),fval,Yhat(i,j),exitflag);
        if exitflag~=1
          Xhat(i,j)=Yhat(i,j);
        end
      end
    end
  elseif scheme==7
    for i=1:n
      for j=1:nRecovd
        [Xhat(i,j),fval,exitflag,output]...
          =fzero(@(x)poly4(x)-Yhat(i,j),Yhat(i,j),optimset('display','off'));
        fprintf('  (%f,%f),Yhat=%f,exitflag=%d\n',...
          Xhat(i,j),fval,Yhat(i,j),exitflag);
        if exitflag~=1
          Xhat(i,j)=Yhat(i,j);
        end
      end
    end
  elseif scheme==8
    for i=1:n
      for j=1:nRecovd
        [Xhat(i,j),fval,exitflag,output]...
          =fzero(@(x)poly7(x)-Yhat(i,j),Yhat(i,j),optimset('display','off'));
        fprintf('  (%f,%f),Yhat=%f,exitflag=%d\n',...
          Xhat(i,j),fval,Yhat(i,j),exitflag);
        if exitflag~=1
          Xhat(i,j)=Yhat(i,j);
        end
      end
    end  
  end

  % recovery rate as defined in sang12effective
  Xref=X(:,(nLeaked+1):(nLeaked+nRecovd));
  relerr=abs((Xref-Xhat)./Xref);
  recovrate(1)=sum(sum(relerr<=epsilon))/(n*nRecovd);
  mean_recovrate(1)=mean_recovrate(1) + recovrate(1);
  
  % recovery rate defined by me
  recovrate(2)=0;
  for j=1:nRecovd
    relerr=abs(X(:,nLeaked+j)-Xhat(:,j))/abs(X(:,nLeaked+j));
    if relerr<=epsilon
      recovrate(2)=recovrate(2)+1;
    end
  end
  recovrate(2)=recovrate(2)/nRecovd;
  mean_recovrate(2)=mean_recovrate(2) + recovrate(2);
  
  % display recovery rates
  fprintf('run %d: %.2f, %.2f\n',sim,recovrate(1),recovrate(2));
end
mean_recovrate=mean_recovrate./nRuns;
fprintf('\nds=%d,sch=%d,att=%d: %.2f, %.2f\n',dataset,scheme,attack,mean_recovrate(1),mean_recovrate(2));

% Results:
