% performs stochastic gradient descent

function [theta, J_hist] = miniBatchGradientDescent(X_norm, y, theta, alpha, iterations, batch_size)

  m = length(y);
  J_hist = zeros(iterations);

  for current = 1:iterations

    [X_batch, y_batch] = miniBatch(X_norm, y, batch_size);
    theta = theta - alpha * (X_batch' * (X_batch * theta - y_batch)) / m;
    J_hist(current) = computeCostMulti(X_norm, y, theta);

  end

end