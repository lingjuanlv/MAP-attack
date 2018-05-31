function pdf=attAMPwMAP_obf(arg)
global pdfT pdfY
if isempty(pdfT)
  error('pdfT not initialized');
end
if isempty(pdfY)
  error('pdfY not initialized');
end
%fprintf('-%d,%d-',size(x,1),size(x,2));
pdf=-evaluate(pdfT,arg())*evaluate(pdfY,arg());
end

