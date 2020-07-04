% retrieves a batch of data points from given samples

function [X_batch, y_batch] = miniBatch(X_norm, y, batch_size)

  m = length(X_norm);
  rows = randperm(m, batch_size);
  X_batch = X_norm(rows, :);
  y_batch = y(rows, :);

end