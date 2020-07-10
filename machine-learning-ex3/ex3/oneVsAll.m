% trains multiple logistic regression classifiers and returns a matrix all_theta
% the i-th row of all_theta corresponds to the classifier for label i

function [all_theta] = oneVsAll(X, y, num_labels, lambda)

	m = size(X, 1);
	n = size(X, 2);

	% weights to return based on the training set
	all_theta = zeros(num_labels, n + 1);

	% add ones to the X data matrix
	X = [ones(m, 1) X];

	for current = 1:num_labels
		% set up the initial theta and options for fmincg
		initial_theta = zeros(n + 1, 1);
		options = optimset('GradObj', 'on', 'MaxIter', 50);

		% optimize wherever y == current label to find theta for
		[theta] = fmincg(@(t)(lrCostFunction(t, X, (y == current), lambda)), initial_theta, options);

		% update the return value
		all_theta(current, :) = theta;
	end

end
