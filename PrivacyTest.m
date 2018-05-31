% in case of problem, "clear all" 
classdef PrivacyTest
  
  properties (Access=private)
    scheme         % 0=random projection
                   % 1=random transformation
                   % 2=tanh + random transformation
                   % 3=double logistic + random transformation
                   % 4=repeated double logistic + random transformation
                   % 5=staircase + random transformation
                   % 6=mixed tanh and repeated double logistic + random transformation
                   % 7=4th-order polynomial
                   % 8=7th-order polynomial
                   % 9=two_Gompertz+uniform random matrix
    attack         % 0=formula
                   % 1=MAP estimation
    fRecovOut      % 0=recovers normal points; 1=recover outliers
    
    pcOutliers=0.1;% fraction of records that are outliers
    pcLeaked=0.2;  % fraction of records leaked    
    pcRecovd=0.1;  % fraction of records to be recovered
                   % in full simulations, set this to (m-nLeaked)
  end
  
  properties (Access=public)
    epsilons=[0.20 0.15 0.10 0.05]; % 0.05 used to be 0.5, leading to confusion
  end
  
  methods (Static)
    % Generates a matrix with extremal values uniformally distributed
    % between 0 and delta or between (1-delta) and 1
    function X = genExtremal(nrows, ncols)
      delta=0.05;

      X=zeros(nrows,ncols);
      for i=1:nrows
        for j=1:ncols
          tmp=randi([0,1]);
          if tmp==0
            tmp=delta*rand();
          else
            tmp=delta*rand()+(1-delta);
          end
          X(i,j)=tmp;
        end
      end
    end
    
    % Perturbs or randomises data matrix using scheme specified by 'scheme'.
    % X should be an n-by-m matrix, where n is the number of attributes, m is the number of records.
    % Y is the nonlinearly perturnbed X.
    % Z is the linearly projected Y.
    % nlfunc is the nonlinear perturbation function, if invNonlin does not exists; otherwise 0.
    % inlfunc is the inverse of the nonlinear perturbation function, if it exists; otherwise 0.
    function [nlfunc, T, inlfunc] = schemeComponents(n, scheme, deltaw)      
      w = floor((n+1)/2) + deltaw;
      if w<=1
        error('schemeComponents: w is too low');
      end
      
      if scheme==0     % random perturbation
        nlfunc = @(x) x;
        T=2*randn(w,n);% sang12effective: s.d. of T's elements has no impact
        inlfunc = @(x) x;
      elseif scheme==1 % random transformation
        nlfunc = @(x) x;
        T=rand(w,n);
        inlfunc = @(x) x;
      elseif scheme==2 % tanh + random transformation
        beta=1.23;
        nlfunc = @(x) tanh(beta*x);
        T=rand(w,n);
        inlfunc = @(x) atanh(x)/beta;
      elseif scheme==3 % double logistic + random transformation
        beta=2.81;
        nlfunc = @(x) sign(x).*(1-exp(-beta*x.^2));
        T=rand(w,n);        
        inlfunc = @(x) sign(x).*((1/beta)*log(1./(1-abs(x)))).^(1/2);
      elseif scheme==4 % repeated double logistic + random transformation
        beta=16.87;
        nlfunc = @(x) sign(x-0.5).*(0.5-0.5*exp(-beta*(x-0.5).^2))+0.5; 
        T=rand(w,n);           
        inlfunc = [];
      elseif scheme==5 % staircase + random transformation
        nlfunc = @staircase;
        T=rand(w,n);        
        inlfunc = @istaircase;
      elseif scheme==6 % mixed tanh and repeated double logistic + random transformation
        nlfunc = @mixedTanhAndRdl;
        T=rand(w,n);        
        inlfunc = [];
      elseif scheme==7 % 4th-order polynomial
        nlfunc = @poly4;
        T=rand(w,n);
        inlfunc = [];
      elseif scheme==8 % 7th-order polynomial
%         nlfunc = @poly7;
        nlfunc = @poly9;
        T=rand(w,n);
        inlfunc = [];
      elseif scheme==9 % two_Gompertz
           nlfunc =@two_Gompertz;
           T=rand(w,n);
           inlfunc = [];
      end
    end    
  end
  
  methods
    % Constructor
    function this = PrivacyTest_new(scheme, attack, fRecovOut)      
      this.scheme = scheme;
      this.attack = attack;
      this.fRecovOut = fRecovOut;
    end

    % X:      Original data matrix
    % deltaw: See schemeComponents  
    function recovrate = do(this, X, deltaw)
      % global variable
      global obf_pdf;
      
      % init local variables
      n = size(X,1);                      % number of attributes (number of rows of X)
      m = size(X,2);                      % number of data records
      nOutliers = m*this.pcOutliers;      % number of leaked records
      nLeaked = m*this.pcLeaked;          % number of outliers to generate
      nRecovd = m*this.pcRecovd;          % number of records to recover
      recovrate = zeros(length(this.epsilons),3); % column 1 has recovery rate defined by Sang et al.
                                                  % column 2 has recovery rate defined by law 
                                                  % column 3 has recovery rate defined by me

      if this.fRecovOut
        % generate and append outliers
        X=[X, this.genExtremal(n,nOutliers)];
        % shuffle outliers to just after leaked columns
        X=[X(:,1:nLeaked), X(:,(m+1):(m+nOutliers)), X(:,(nLeaked+1):m)];
      end

      %- - - - - - - -Perturb data- - - - - - - -
      [nlfunc, T, inlfunc] = this.schemeComponents(n, this.scheme, deltaw);
      Y=nlfunc(X);
      Z=T*Y;

      % figure(77)
      % subplot(2,2,1); plot(X(1,:),X(2,:),'+');
      % subplot(2,2,2); plot(X(1,:),X(3,:),'+');
      % subplot(2,2,3); plot(X(2,:),X(3,:),'+');
      % subplot(2,2,4); plot(Z(1,:),Z(2,:),'+');

      %- - - - - - - -Attack- - - - - - - -
      % init data structures
      Xhat=zeros(size(X,1), nRecovd);
      Yhat=zeros(size(Y,1), nRecovd); 

      % attack
      if this.attack==0
        ybar=mean(Y(:,1:nLeaked),2);
        sigmay=cov(Y(:,1:nLeaked)');
        covZ=cov(Z'); 
        for j=1:nRecovd
          fprintf('%d..',j);
          z=Z(:,nLeaked+j);

          Yhat(:,j)=ybar + sigmay*T'*(covZ\(z-T*ybar));
        end
      elseif this.attack==1
        % estimate data pdf based on leaked data; obf_pdf used in attMAP_obf
        obf_pdf=kde(Y(:,1:nLeaked),'lcv',ones(1,nLeaked),'Epan');

        % the first nLeaked records are leaked
        % the next nRecovd records are to be recovered
        for j=1:nRecovd
          fprintf('%d..',j);
          z=Z(:,nLeaked+j);

          x0=mean(Y(:,1:nLeaked),2);
          opts=optimset('Algorithm','interior-point');
          problem=createOptimProblem('fmincon','objective',@estMAP_obf,...
            'x0',x0,'Aeq',T,'beq',z,'lb',zeros(n,1),'ub',ones(n,1),'options',opts);
          gs=GlobalSearch; % MultiStart is not better
          Yhat(:,j)=run(gs,problem);
        end
      end

      % invert the nonlinear function
      if ~isempty(inlfunc)
        Xhat=inlfunc(Yhat);
      else
        for i=1:n
          for j=1:nRecovd
            [Xhat(i,j),fval,exitflag,~]...
              =fzero(@(x) nlfunc(x)-Yhat(i,j),Yhat(i,j),optimset('display','off'));
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
      for i=1:length(this.epsilons)
        epsilon=this.epsilons(i);
        recovrate(i,1)=sum(sum(relerr<=epsilon))/(n*nRecovd);
      end

      % recovery rate defined by law
      for j=1:nRecovd  
        relerr=norm(Xref(:,j)-Xhat(:,j))/norm(Xref(:,j));
        %fprintf('relerr=%.4f\n',relerr);
        for i=1:length(this.epsilons)
          if relerr<=this.epsilons(i)
            recovrate(i,2)=recovrate(i,2)+1;
          end
        end
      end
      recovrate(:,2)=recovrate(:,2)./nRecovd;
      
      % recovery rate defined by me
      for j=1:nRecovd  
        relerr=1-abs(dot(Xref(:,j)-mean(Xref,2),Xhat(:,j)-mean(Xhat,2)))/(norm(Xref(:,j)-mean(Xref,2))*norm(Xhat(:,j)-mean(Xhat,2)));
        %fprintf('relerr=%.4f\n',relerr);
        for i=1:length(this.epsilons)
          if relerr<=this.epsilons(i)
            recovrate(i,3)=recovrate(i,3)+1;
          end
        end
      end
      recovrate(:,3)=recovrate(:,3)./nRecovd;
    end
  end
end