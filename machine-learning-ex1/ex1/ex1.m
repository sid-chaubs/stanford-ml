% sample run of gradient descent for univariate linear regression

clear;
close all;
clc;

warmUpExercise();

data = load('ex1data1.txt');

X = data(:, 1);
y = data(:, 2);

X = [ones(length(y), 1), data(:,1)];

% initialize fitting parameters
theta = zeros(2, 1);

% max. number of iterations in gradient descent
iterations = 1500;

% learning rate per iteration
alpha = 0.01;

theta = gradientDescent(X, y, theta, alpha, iterations);
fprintf('Expected theta values (approx): -3.6303 1.1664. Result: %f %f \n', theta(1), theta(2));