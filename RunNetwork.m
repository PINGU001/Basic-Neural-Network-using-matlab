function [v1,y1,v2,y2,v3,y3,v4,y4,v5,y5,v6,y6]=RunNetwork(W1,W2,W3,W4,W5,W6,X)
% W1, W2, W3, W4, W5, W6  :   Weights of neural system
% x           :   Values of input neurons

v1 = W1*X;
y1 = sigmoid(v1);
v2 = W2*y1;
y2 = sigmoid(v2);
v3 = W3*y2;
y3 = sigmoid(v3);
v4 = W4*y3;
y4 = sigmoid(v4);
v5 = W5*y4;
y5 = sigmoid(v5);
v6 = W6*y5;
y6 = sigmoid(v6);

end