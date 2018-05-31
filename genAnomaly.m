function anomaly=genAnomaly(testData)
NoAno=round(size(testData,1)/20);
anomaly=rand(NoAno,size(testData,2));
%inputs=[D,anomaly];

