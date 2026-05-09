clc; clear; close all;

%% ==========================================
% PARAMETROS DEFINIDOS POR EL USUARIO
% ==========================================
N = 8;                    % numero de nodos
n = 3;                    % dimension de cada nodo
Tspan = [0 30];           % tiempo de simulacion
c = 1.5;                  % ganancia global de acoplamiento

topology = 'exponential';  % 'ring', 'regular', 'smallworld', 'random', 'exponential'
nonlinear_coupling = true;
time_varying_weights = true;

% Tipos de nodos:
% 1 = Lorenz
% 2 = Rossler
nodeTypes = [1 1 1 2 2 1 2 1];

if length(nodeTypes) ~= N
    error('La longitud de nodeTypes debe ser igual a N');
end

%% Matriz de acoplamiento interno
Gamma = diag([1.0, 0.5, 0.2]);

%% Parametros de topologia
paramsTopo.k = 2;         
paramsTopo.p = 0.25;      
paramsTopo.pr = 0.30;     
paramsTopo.lambda = 0.7;  

%% Construir topologia
A = build_topology(N, topology, paramsTopo);

% Asegurar red no dirigida
A = double(A ~= 0);
A = max(A, A');
A(1:N+1:end) = 0;

%% Construir matriz de pesos
% Pesos aleatorios solo donde hay conexion
R = 0.5 + rand(N);
W0 = A .* R;

% Hacer W0 simetrica para red no dirigida
W0 = (W0 + W0')/2;
W0 = W0 .* A;
W0(1:N+1:end) = 0;

%% Dibujar red con pesos
plot_network_weights(A, W0, topology, nodeTypes);

%% Mostrar matriz de adyacencia
figure('Color','w');
imagesc(A);
colorbar;
axis square;
xlabel('Nodo j');
ylabel('Nodo i');
title('Matriz de adyacencia A');

%% Mostrar matriz de pesos
figure('Color','w');
imagesc(W0);
colorbar;
axis square;
xlabel('Nodo j');
ylabel('Nodo i');
title('Matriz de pesos W0');

%% Condiciones iniciales
X0 = zeros(N*n,1);
for i = 1:N
    X0((i-1)*n+1:i*n) = 5*randn(n,1);
end

%% Parametros para simulacion
params.N = N;
params.n = n;
params.c = c;
params.A = A;
params.W0 = W0;
params.Gamma = Gamma;
params.nodeTypes = nodeTypes;
params.nonlinear_coupling = nonlinear_coupling;
params.time_varying_weights = time_varying_weights;

%% Simulacion
[t,X] = ode45(@(t,x) network_rhs(t,x,params), Tspan, X0);

%% Graficas temporales
plot_results(t, X, params, 'Sin control');

%% Error de sincronizacion
E = sync_error(X, N, n);
figure('Color','w');
plot(t, E, 'LineWidth', 2);
grid on;
xlabel('Tiempo');
ylabel('Error promedio de sincronizacion');
title(['Error de sincronizacion - Topologia: ', topology]);

%% Retrato de fase del nodo 1
node_id = 1;
x1 = X(:, (node_id-1)*n + 1);
x2 = X(:, (node_id-1)*n + 2);
x3 = X(:, (node_id-1)*n + 3);

figure('Color','w');
plot3(x1, x2, x3, 'LineWidth', 1.2);
grid on;
xlabel('x');
ylabel('y');
zlabel('z');
title(['Retrato de fase 3D del nodo ', num2str(node_id)]);