% plots data points X and y into a new figure

function plotData(X, y)

	figure;
	hold on;

	posIdx = find(y == 1);
	negIdx = find(y == 0);

	plot(X(posIdx, 1), X(posIdx, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
	plot(X(negIdx, 1), X(negIdx, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

	hold off;

end