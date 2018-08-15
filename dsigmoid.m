function y=dsigmoid(x)
% The derivative of the sigmoid function

y = sigmoid(x).*(1-sigmoid(x));

end