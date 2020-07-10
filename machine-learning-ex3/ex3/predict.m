% predict the label of an input given a trained neural network

function p = predict(Theta1, Theta2, X)
	m = size(X, 1);
	num_labels = size(Theta2, 1);

	p = zeros(size(X, 1), 1);

	% add an intercept term
	X = [ones(m, 1) X];

	% add intercept term to the output of the first layer
	Layer1 = [ones(m, 1) sigmoid(X * Theta1')];

	% add intercept term to the output of the first layer
	probabilities = [sigmoid(Layer1 * Theta2')];

	for i = 1:m
		[val idx] = max(probabilities(i, 1:end));
		p(i) = idx;
	end

end
