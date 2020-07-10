% predict the label for a trained one-vs-all classifier

function p = predictOneVsAll(all_theta, X)

	m = size(X, 1);
	num_labels = size(all_theta, 1);

	% predictions - the digit corresponding to the max scores for the trained values of theta
	p = zeros(m, 1);
	X = [ones(m, 1) X];

	probabilities = sigmoid(X * all_theta');

	for i = 1:m
		[val num] = max(probabilities(i, 1:end));
		p(i) = num;
	end

end
