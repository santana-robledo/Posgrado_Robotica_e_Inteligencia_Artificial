function plot_results_with_ref(t, X, X_ref, params, figTitle)
    N = params.N; n = params.n;
    figure('Color', 'w', 'Name', figTitle);
    for dim = 1:n
        subplot(n,1,dim); hold on;
        plot(t, X_ref(:, dim), 'k--', 'LineWidth', 2, 'DisplayName', 'REF'); % Referencia
        for i = 1:N
            plot(t, X(:, (i-1)*n + dim), 'LineWidth', 1);
        end
        grid on; ylabel(['x_', num2str(dim)]);
    end
    xlabel('Tiempo'); title(subplot(n,1,1), figTitle);
end