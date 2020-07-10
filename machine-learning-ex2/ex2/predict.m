% predicts whether the label is 0 or 1 using learned logistic

function p = predict(theta, X)

	threshold = 0.5;
	hypothesis = arrayfun(@sigmoid, X * theta);
	p = zeros(length(X), 1);
	p(find(hypothesis >= threshold)) = 1;

end
