function train_x=mix_anomaly(X,nrows,ncols,NoAno) %X:training sata,row:records,column:attribute
    
  
    % Generates a matrix with extremal values uniformally distributed
    % between 0 and delta or between (1-delta) and 1
    function X = genAnomaly(nrows, ncols)
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
    
    % X should be an n-by-m matrix, where n is the number of attributes, m is the number of records.
    % Y is the nonlinearly perturnbed X.
    % Z is the linearly projected Y.
    % nlfunc is the nonlinear perturbation function, if invNonlin does not exists; otherwise 0.
    % inlfunc is the inverse of the nonlinear perturbation function, if it exists; otherwise 0.
      
      % init local variables
      n = size(X,1);                      % number of records
      m = size(X,2);                      % number of attributes

        % generate and append outliers
%         X=[X; genAnomaly(nOutliers,m)];
          train_x=[X; genAnomaly(NoAno,m)];
        % shuffle outliers to just after leaked columns
%         train_x=[X(1:nLeaked,:); X((m+1):(m+nOutliers),:); X((nLeaked+1):m,:)];
end