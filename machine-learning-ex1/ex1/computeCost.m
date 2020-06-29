% compute cost for univariate linear regression

function J = computeCost(X, y, theta)

	% compute line
	fn = theta(1) + theta(2) .* X(:, 2);

	% compute cost
  J = sum((fn - y(:)) .^ 2) / (2 * length(y));

end