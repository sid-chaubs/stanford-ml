% copmutes the closed-form solution to linear regression

function [theta] = normalEqn(X, y)

	theta = pinv(X' * X) * X' * y;

end
