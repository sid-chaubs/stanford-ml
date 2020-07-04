% compute cost and gradient for logistic regression

function [J, grad] = costFunction(theta, X, y)

	m = length(y);

	% hypothesis function
	hx = arrayfun(@sigmoid, X * theta);

	% cost function
	J =  sum(-y .* log(hx) - (1 - y) .* log(1 - hx)) / m;

	% gradient
	grad = sum((hx - y) .* X) / m;

end
