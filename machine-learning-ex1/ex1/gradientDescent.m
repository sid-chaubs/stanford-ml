% performs gradient descent

function theta = gradientDescent(X, y, theta, alpha, iterations)

  m = length(y);

  for current = 1:iterations
    % new theta = theta - (learning rate * derivative of cost function) / m;
    theta = theta - alpha * (X' * (X * theta - y)) / m;
  end

end