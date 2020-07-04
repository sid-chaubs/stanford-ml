% computes sigmoid function value

function g = sigmoid(x)

	g = e .^ x / (1 + e .^ x);

end