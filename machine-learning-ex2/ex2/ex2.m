% test run of gradient descent for univariate logistic regression

clear;
close all;
clc;

data = load('ex2data1.txt');

X = data(:, [1, 2]); y = data(:, 3);
[m, n] = size(X);
X = [ones(m, 1) X];
initial_theta = zeros(n + 1, 1);

% compute cost with fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

fprintf('cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf('%f\n', theta);

prob = sigmoid([1 45 85] * theta);
fprintf('For a student with scores 45 and 85, we predict an admission probability of %f\n', prob);
fprintf('Expected value: 0.775 +/- 0.002\n\n');

p = predict(theta, X);
fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('Expected accuracy (approx): 89.0\n');