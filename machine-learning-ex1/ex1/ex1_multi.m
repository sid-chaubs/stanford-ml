% sample run of gradient descent for multivariate linear regression

clear;
close all;
clc;

data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

alpha = 0.1;
theta = zeros(3, 1);
[X_norm mu sigma] = featureNormalize(X);
X_norm = [ones(m, 1) X_norm];

% run gradient descent
num_iters = 1500;
[theta, J_history] = gradientDescentMulti(X_norm, y, theta, alpha, num_iters);
fprintf('Theta computed from gradient descent: \n');
display(theta);

% run mini batch gradient descent
num_iters = 10000;
[theta, J_history] = miniBatchGradientDescent(X_norm, y, theta, alpha, num_iters, 10);
fprintf('Theta computed from stochastic gradient descent: \n');
display(theta);

% run stochastic gradient descent
num_iters = 10000;
[theta, J_history] = stochasticGradientDescent(X_norm, y, theta, alpha, num_iters);
fprintf('Theta computed from stochastic gradient descent: \n');
display(theta);

% predicted price for a 1650 sq-ft, 3 br house based on gradient descent results
price = makePrediction([1650 3], mu, sigma, theta);

fprintf(['Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f\n'], price);

% Calculate the parameters from the normal equation
theta = normalEqn([ones(m, 1) X], y);

% estimate the price of a 1650 sq-ft, 3 br house based on results from normal equation
price = [1 1650 3] * theta;

fprintf(['Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n $%f\n'], price);