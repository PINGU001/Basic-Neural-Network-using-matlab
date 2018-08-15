function [deltaW1,deltaW2,deltaW3,deltaW4,deltaW5,deltaW6] = BackPropagationLearn(W1,W2,W3,W4,W5,W6,X,d,alpha)

% W1,W2,W3 are the wieght matrixes
% X is the input for the entire system (image)
% d is the training data (numerof blobs)
% alpha is the learnig rate

% y is the output of each layer after the sigmiod functoin is applied
% v is the output of each layer respectivly before the sigmoid function is apllied to it
[v1,y1,v2,y2,v3,y3,v4,y4,v5,y5,v6,y6] = RunNetwork(W1,W2,W3,W4,W5,W6,X);

E6 = d - y6;

[deltaW6,E5] = BackPropagationStep(W6,y5,E6,v6,alpha);
[deltaW5,E4] = BackPropagationStep(W5,y4,E5,v5,alpha);
[deltaW4,E3] = BackPropagationStep(W4,y3,E4,v4,alpha);
[deltaW3,E2] = BackPropagationStep(W3,y2,E3,v3,alpha);
[deltaW2,E1] = BackPropagationStep(W2,y1,E2,v2,alpha);
[deltaW1,E0] = BackPropagationStep(W1,X,E1,v1,alpha);


end