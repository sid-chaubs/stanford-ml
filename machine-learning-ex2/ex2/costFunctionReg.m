% applies regularization based on the lambda paramter to compute cost and gradient for logistic regression

function [J, grad] = costFunctionReg(theta, X, y, lambda)

	m = length(y);
	grad = zeros(size(theta));
	J = 0;

	% hypothesis function
	h = sigmoid(X * theta);

	% theta for regularization
	theta_reg = theta(2:end);

	% cost function
	J = (-y' * log(h) - (1 - y)' * log((1 - h))) / m + sum(theta_reg .^ 2) * lambda / (2 * m);

	% gradient
	grad = (1 / m) * X' * (h - y);
	grad(2:end) += (lambda / m) * theta_reg;

end
