% performs stochastic gradient descent

function [theta, J_hist] = stochasticGradientDescent(X_norm, y, theta, alpha, iterations)

  m = length(y);
  J_hist = zeros(iterations);

  for current = 1:iterations

    % get sample data point
    sample = randi(m);
    X_sample = X_norm(sample, :);
    Y_sample = y(sample);

    theta = theta - alpha * (X_sample' * (X_sample * theta - Y_sample)) / m;
    J_hist(current) = computeCostMulti(X_norm, y, theta);

  end

end