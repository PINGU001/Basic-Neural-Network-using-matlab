function [deltaW,backE]=BackPropagationStep(W,X,E,V,alpha)
% Computes basic step of back propagation.
% W   - Weights of layer
% X   - Input of layer
% E   - Error at output of layer
% V   - Output of layer as computed by layer (redundant; V=W*X)
% alpha - Learning rate

 

deltaW = (alpha*X*(dsigmoid(V).*E)')';
backE = W'*E;

end