% test run of logistic regression with regularization

clear;
close all;
clc;

% load data
data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);
X = mapFeature(X(:,1), X(:,2));

% set up initial weights and regularization parameter
initial_theta = zeros(size(X, 2), 1);
lambda = 1;

% configure options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

% compute accuracy on our training set
p = predict(theta, X);

fprintf('Training accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('Expected accuracy (with lambda = 1): 83.1 (approx)\n');