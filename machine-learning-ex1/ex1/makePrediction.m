% returns house price based on results of gradient descent

function prediction = makePrediction(input, mu, sigma, theta)

	prediction = 0;
	normalized = zeros(size(input));

	for i = 1:columns(input)
	 	normalized(:, i) = (input(:, i) - mu(i)) / sigma(i);
	end

	prediction = [1 normalized] * theta;

end