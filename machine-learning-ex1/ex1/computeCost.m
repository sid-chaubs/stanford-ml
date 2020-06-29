% uses mean squared error function in to compute the cost of using theta as the parameter for linear regression to fit the data points in X and y.
% compute cost for linear regression with one variable

function J = computeCost(X, y, theta)

	J = 0;
	m = length(X);

	for i = 1: m
		% straight line equation
		hx = theta(1) + theta(2) * X(i, 2);

		% mean squared error = (1 / 2m) * (h(xi) - yi)^2 ...(where i = 1...n)
		J = J + (hx - y(i)) ^ 2 / (2 * m);
	end

end