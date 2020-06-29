 % feature normalization function

function [X_norm, mu, sigma] = featureNormalize(X)

	mu = mean(X);
	sigma = std(X);
	m = length(X);

	X_norm = zeros(size(X));

	for i = 1:columns(X)
	 	X_norm(:, i) = (X(:, i) - mu(i)) / sigma(i);
	end

end
