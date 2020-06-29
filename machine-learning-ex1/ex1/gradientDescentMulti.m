% performs gradient descent for multivariate linear regression

function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

    m = length(y);
    J_history = zeros(num_iters, 1);

    for iter = 1:num_iters
    	% coefficients = coefficients - (learning rate) * (derivative of cost function)
      theta = theta - alpha * (X' * (X * theta - y)) / m;
      J_history(iter) = computeCostMulti(X, y, theta);
    end

end
