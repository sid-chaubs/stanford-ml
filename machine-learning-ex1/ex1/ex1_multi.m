% sample run of gradient descent for multivariate linear regression

clear;
close all;
clc;

data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% max iterations for gradient descent
num_iters = 2500;
% learning rate per iteration
alpha = 0.01;
% initialize fitting parameters
theta = zeros(3, 1);

% normalization and add intercept
[X_norm mu sigma] = featureNormalize(X);
X_norm = [ones(m, 1) X_norm];

% run gradient descent
[theta, J_history] = gradientDescentMulti(X_norm, y, theta, alpha, num_iters);

fprintf('Theta computed from gradient descent: %f \n', theta);

% predicted price for a 1650 sq-ft, 3 br house based on gradient descent results
price = makePrediction([1650 3], mu, sigma, theta);

fprintf(['Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f\n'], price);

% Calculate the parameters from the normal equation
theta = normalEqn([ones(m, 1) X], y);

% estimate the price of a 1650 sq-ft, 3 br house based on results from normal equation
price = [1 1650 3] * theta;

fprintf(['Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n $%f\n'], price);