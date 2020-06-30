% performs gradient descent

function [theta, J_hist] = gradientDescent(X, y, theta, alpha, iterations)

  m = length(y);
  J_hist = zeros(iterations);

  for current = 1:iterations
    J_hist(current) = computeCost(X, y, theta);
    theta = theta - alpha * (X' * (X * theta - y)) / m;
  end

end