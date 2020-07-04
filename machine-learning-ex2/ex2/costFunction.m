% compute cost and gradient for logistic regression

function [J, grad] = costFunction(theta, X, y)

	m = length(y);

	% current hypothesis
	h = sigmoid(X * theta);

	% cost function
	J = sum(-y' * log(h) - (1 - y)' * log(1 - h)) / m;

	% gradient
	grad = sum((h - y)' * X) / m;

end