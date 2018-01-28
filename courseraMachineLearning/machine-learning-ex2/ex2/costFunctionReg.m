function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%sigmoid predicts what the output is going to be
h = sigmoid(X*theta);
%========calculate cost===========
%calculate the cost, which will be the same as previously
cost = sum((-y .* log(h)) - (1-y) .* log(1-h));
%must regularize the cost by addint sum of theta^2 from index 2 -> end
J = cost/m + (lambda/(2.0*m)) * sum(theta(2:size(theta)).^2);

%========calculate gradient=======

%compute the regular gradient
grad = X' * (h-y);
%must account for added terms, minimize the added parameters
reg = lambda * theta;
%set the first element of reg to 0, because we don't want to add
%for theta(0)
reg(1) = 0;
%add gradient and reg together
grad = grad + reg;
%divid by m
grad = grad ./ m;


% =============================================================

end
