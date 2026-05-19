clc; clear; close all;

%% 1. Configuración de Parámetros
N = 8;                    % numero de nodos
n = 3;                    % dimension de cada nodo
Tspan = [0 30];           % tiempo de simulacion
c = 1.5;                  % ganancia global de acoplamiento

topology = 'exponential';  % 'ring', 'regular', 'smallworld', 'random', 'exponential'
nonlinear_coupling = true; 
time_varying_weights = true;
control_system = false;     

% Tipos de nodos: 1 = Lorenz, 2 = Rossler
nodeTypes = [1 2 1 2 1 2 1 2];

% Parámetros de la Referencia
params.ref_type = 1;       % La referencia será un Lorenz
x_ref0 = [0.5; 0.5; 0.5];  % Estado inicial de la referencia

%% 2. Matriz de acoplamiento interno
Gamma = diag([1.0, 0.5, 0.2]);

%% 3. Parámetros de topología y construcción
paramsTopo.k = 2;         
paramsTopo.p = 0.25;      
paramsTopo.pr = 0.30;     
paramsTopo.lambda = 0.7;  

A = build_topology(N, topology, paramsTopo);

% Construir matriz de pesos inicial W0
R = 0.5 + rand(N);
W0 = A .* R; 
W0 = (W0 + W0')/2;
W0(1:N+1:end) = 0;

%% 4. Condiciones iniciales de los Nodos
X0 = zeros(N*n,1);
for i = 1:N
    X0((i-1)*n+1:i*n) = 5*randn(n,1);
end

% Vector de estado total: [Nodos; Referencia]
X0_total = [X0; x_ref0]; 

%% 5. Empaquetado de parámetros para ODE45
params.N = N;
params.n = n;
params.c = c;
params.A = A;
params.W0 = W0;
params.Gamma = Gamma;
params.nodeTypes = nodeTypes;
params.nonlinear_coupling = nonlinear_coupling;
params.time_varying_weights = time_varying_weights;
params.control_system = control_system; 
params.K_gain = 15; % Ganancia de control (Ajustada para mayor precisión)

%% 6. Simulación
[t, X_total] = ode45(@(t,x) network_rhs(t,x,params), Tspan, X0_total);

% Separar resultados:
X_nodes = X_total(:, 1:N*n);          % Estados de los 8 nodos
X_ref = X_total(:, N*n+1:end);        % Estado de la referencia (3 variables)

%% 7. Graficación de Resultados
% Dibujar red
plot_network_weights(A, W0, topology, nodeTypes);

% Gráficas temporales (Nodos vs Referencia)
plot_results_with_ref(t, X_nodes, X_ref, params, 'Control hacia Referencia');

% Error de sincronización respecto a la REFERENCIA
E_ref = zeros(length(t), 1);
for i = 1:N
    idx = (i-1)*n + 1;
    diff = X_nodes(:, idx:idx+2) - X_ref;
    E_ref = E_ref + sqrt(sum(diff.^2, 2));
end
E_ref = E_ref / N;

figure('Color','w');
plot(t, E_ref, 'r', 'LineWidth', 2);
grid on;
xlabel('Tiempo');
ylabel('Error promedio respecto a referencia');
title('Convergencia del Sistema de Control');

% Retrato de fase del nodo 1 vs Referencia
node_id = 1;
idx = (node_id-1)*n;
figure('Color','w');
plot3(X_nodes(:, idx+1), X_nodes(:, idx+2), X_nodes(:, idx+3), 'b', 'LineWidth', 1);
hold on;
plot3(X_ref(:,1), X_ref(:,2), X_ref(:,3), 'r--', 'LineWidth', 1.5);
grid on;
legend(['Nodo ', num2str(node_id)], 'Referencia');
title('Retrato de fase: Nodo vs Referencia');

%Matriz de Adyacencia
figure('Color','w', 'Name', 'Matriz de Adyacencia');
imagesc(A); 
colormap(gca, [1 1 1; 0.9 0.9 0.9]); % Fondo casi blanco para legibilidad
axis square;
title('Matriz de Adyacencia (A)');
xlabel('Nodo j'); ylabel('Nodo i');
set(gca, 'XTick', 1:N, 'YTick', 1:N);

% Añadir los números de la matriz A
for i = 1:N
    for j = 1:N
        text(j, i, num2str(A(i,j)), ...
            'HorizontalAlignment', 'center', ...
            'FontSize', 12, 'FontWeight', 'bold');
    end
end

% 2. Matriz de Pesos (W0)
figure('Color','w', 'Name', 'Matriz de Pesos');
imagesc(W0); 
colormap(gca, white); % Fondo blanco puro
axis square;
title('Matriz de Pesos (W_0)');
xlabel('Nodo j'); ylabel('Nodo i');
set(gca, 'XTick', 1:N, 'YTick', 1:N);

% Añadir los números de los pesos (redondeados a 2 decimales)
for i = 1:N
    for j = 1:N
        if W0(i,j) ~= 0
            valStr = num2str(W0(i,j), '%.2f');
            text(j, i, valStr, ...
                'HorizontalAlignment', 'center', ...
                'FontSize', 10, 'Color', 'k');
        else
            text(j, i, '0', 'HorizontalAlignment', 'center', 'Color', [0.7 0.7 0.7]);
        end
    end
end
