% computes cost and gradient for logistic regression with regularization

function [J, grad] = lrCostFunction(theta, X, y, lambda)

	m = length(y);
	hx = sigmoid(X * theta);

	% compute cost
	J = (1 / m) * sum(-y' * log(hx) - (1 - y)' * log(1 - hx)) + lambda * sum(theta(2:end) .^ 2) / (2 * m);

	% compute gradient
	grad = (1 / m) * (X' * (hx - y));

	% account for regularization
	grad(2:end) += (lambda / m) * theta(2:end);

end
