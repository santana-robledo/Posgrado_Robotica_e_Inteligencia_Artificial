function plot_results(t, X, params, figTitle)

N = params.N;
n = params.n;

figure('Name', figTitle, 'Color', 'w');

for dim = 1:n
    subplot(n,1,dim); hold on;
    for i = 1:N
        xi = X(:, (i-1)*n + dim);
        plot(t, xi, 'LineWidth', 1.2);
    end
    grid on;
    xlabel('Tiempo');
    ylabel(['x_', num2str(dim)]);
    title([figTitle, ' - Componente ', num2str(dim)]);
end
end