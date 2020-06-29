% compute cost for multivariate linear regression

function J = computeCostMulti(X, y, theta)

	% vectorized form of mean squared error cost function
	J = ((X * theta - y)' * (X * theta - y)) / (2 * length(y));

end
