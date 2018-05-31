function runPrivacyTest(which_test, scheme, attack, fRecovOut)

  % which_test              % a string identifying the test
  % scheme                  % refer to PrivacyTest.m
  % attack                  % refer to PrivacyTest.m
  % fRecovOut               % refer to PrivacyTest.m

  %- - - - - - - -BEGIN configurable section- - - - - - - -
  n=15;
  m=1000;                      % number of records
  nRuns=10;                    % number of simulation runs
%   datapath='../Datasets/';
  datapath='preprocess_datasets/';
  respath='out/';
  %- - - - - - - -END configurable section- - - - - - - -

  % Load Gaussian data, perturb it, and estimate original from perturbed data
  rng(1);

  %- -Select input and output files- -
  if strcmp(which_test,'Gaussian')
    path=strcat(datapath,sprintf('Gaussian-n=%d,sigma=1.0.mat',n));
    outpath=strcat(respath,sprintf('Gaussian-sch=%d,att=%d,ro=%d.mat',...
      scheme,attack,fRecovOut));    
    
    test_range = -5:0;
    
  elseif strcmp(which_test,'Laplace')
    path=strcat(datapath,sprintf('Laplace-n=%d.mat',n));
    outpath=strcat(respath,sprintf('Laplace-sch=%d,att=%d,ro=%d.mat',...
      scheme,attack,fRecovOut));
    
    test_range = -5:0;
    
  else
    path=strcat(datapath,sprintf('%s.mat',which_test));
    outpath=strcat(respath,sprintf('%s-sch=%d,att=%d,ro=%d.mat',...
      which_test,scheme,attack,fRecovOut));
    
    test_range = 0;
  end
    
  %- -Load data and init tests- -
  ds=load(path);
  if size(ds.X,1)<m
    error(strcat('Problem with dataset ',path));
  else
    fprintf('Loaded %d records\n',size(ds.X,1));
  end
  test = PrivacyTest(scheme,attack,fRecovOut);

  % Each cell corresponds to a deltaw, and contains an array
  % (runs_recovrate) of n_epsilonsx2 matrices, so
  % test_recovrate is an array of array of matrices
  test_recovrate = cell(1,length(test_range));

  % Each cell corresponds to a deltaw, and contains an n_epsilonsx2 matrix
  mean_recovrate = cell(1,length(test_range));
  stdv_recovrate = cell(1,length(test_range));

  %- -Run tests- - 
  for i=1:length(test_range)
    deltaw=test_range(i);
    
    runs_recovrate=cell(1,nRuns);
    sum_runs_recovrate=zeros(length(test.epsilons),2);
    sumsq_runs_recovrate=zeros(length(test.epsilons),2);
    
    for run=1:nRuns
      X=(ds.X(randperm(size(ds.X,1),m), :))';
      recovrate = test.do(X,deltaw); % n_epsilons*2 matrix     
      
      % For each run, recovrate is a n_epsilon*2 matrix, where n_epsilon is
      % the number epsilons
      runs_recovrate{run} = recovrate;
      sum_runs_recovrate = sum_runs_recovrate + recovrate;
      sumsq_runs_recovrate = sumsq_runs_recovrate + recovrate.^2;
      
      fprintf('\n  run %2d: (%.2f) %.4f, %.4f, (%.2f) %.4f, %.4f\n', run, ...
        test.epsilons(1), recovrate(1,1), recovrate(1,2), ...        
        test.epsilons(3), recovrate(3,1), recovrate(3,2));
    end

    test_recovrate{i} = runs_recovrate; % an array of n_epsilon*2 matrices
    mean_recovrate{i} = sum_runs_recovrate ./ nRuns;
    stdv_recovrate{i} = sqrt(sumsq_runs_recovrate./nRuns - mean_recovrate{i}.^2);
    fprintf('\n  deltaw=%2d: (%.2f) %.4f<%.4f>, %.4f<%.4f>',deltaw,test.epsilons(1), ...
      mean_recovrate{i}(1,1), stdv_recovrate{i}(1,1), ...
      mean_recovrate{i}(1,2), stdv_recovrate{i}(1,2));
    fprintf('\n  deltaw=%2d: (%.2f) %.4f<%.4f>, %.4f<%.4f>\n\n',deltaw,test.epsilons(3), ...
      mean_recovrate{i}(3,1), stdv_recovrate{i}(3,1), ...
      mean_recovrate{i}(3,2), stdv_recovrate{i}(3,2));
  end

  %- -Output test results- -
  save(outpath,'which_test','scheme','attack','fRecovOut','n','m','nRuns',...
    'test_range','test_recovrate','mean_recovrate','stdv_recovrate');

  fprintf('\nsch=%d,att=%d,recovOut=%d\n',scheme,attack,fRecovOut);
  for i=1:length(test_range)
    fprintf('\n  deltaw=%2d: (%.2f) %.4f<%.4f>, %.4f<%.4f>',test_range(i),test.epsilons(1), ...
      mean_recovrate{i}(1,1), stdv_recovrate{i}(1,1), ...
      mean_recovrate{i}(1,2), stdv_recovrate{i}(1,2));
  end
  fprintf('\n');
  for i=1:length(test_range)
    fprintf('\n  deltaw=%2d: (%.2f) %.4f<%.4f>, %.4f<%.4f>',test_range(i),test.epsilons(2), ...
      mean_recovrate{i}(2,1), stdv_recovrate{i}(2,1), ...
      mean_recovrate{i}(2,2), stdv_recovrate{i}(2,2));
  end
  fprintf('\n');
  for i=1:length(test_range)
    fprintf('\n  deltaw=%2d: (%.2f) %.4f<%.4f>, %.4f<%.4f>',test_range(i),test.epsilons(3), ...
      mean_recovrate{i}(3,1), stdv_recovrate{i}(3,1), ...
      mean_recovrate{i}(3,2), stdv_recovrate{i}(3,2));
  end
  fprintf('\n');
end
