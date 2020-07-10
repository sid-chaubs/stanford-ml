% test script for neural network code

clear;
close all;
clc;

input_layer_size  = 400; % 20x20 Input Images of Digits
hidden_layer_size = 25; % 25 hidden units
num_labels = 10; % 10 labels, from 1 to 10 (note "0" is mapped to label 10)

load('ex3data1.mat');
m = size(X, 1);

% load the weights for the neural net into variables Theta1 and Theta2
load('ex3weights.mat');
pred = predict(Theta1, Theta2, X);

fprintf('Prediction Accuracy: %f\n\n', mean(double(pred == y)) * 100);