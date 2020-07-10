% test script for one-vs-all logistic regression.

clear;
close all;
clc;

input_layer_size = 400;
num_labels = 10;

% load training data, each training example contains a gray scale image of an integer from 0-9.
% digit zero is being mapped to the value ten, others are labeled 1-9
load('ex3data1.mat');

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

% predict the label for a trained one-vs-all classifier
pred = predictOneVsAll(all_theta, X);
fprintf('\nPrediction Accuracy: %f\n', mean(double(pred == y)) * 100);
