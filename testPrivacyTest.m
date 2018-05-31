for scheme=0:8
  [nlfunc,T,inlfunc]=PrivacyTest.schemeComponents(10,scheme);
  if ~isempty(nlfunc) && ~isempty(inlfunc) 
    X=rand(1,100);
    Y=nlfunc(X);
    X2=inlfunc(Y);
    fprintf('%d: norm=%f\n', scheme, norm(X2-X));
  end
end