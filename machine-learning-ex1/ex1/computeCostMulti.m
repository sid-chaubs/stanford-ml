% uses mean squared error function in to compute the cost of using theta as the parameter for linear regression to fit the data points in X and y.
% compute cost for linear regression with multiple variables

function J = computeCostMulti(X, y, theta)

	% vectorized form of mean squared error cost function
	J = ((X * theta - y)' * (X * theta - y)) / (2 * length(y));

end
