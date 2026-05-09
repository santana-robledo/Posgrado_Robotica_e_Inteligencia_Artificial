clc; clear; close all;

% Numero de nodos
N = 4;

% Dimension de cada nodo
n = 3;

% Acoplamiento global
c = 1.0;

% Topologia manual simple (anillo)
A = [0 1 0 1;
    1 0 1 0;
    0 1 0 1;
    1 0 1 0];

% Pesos
W0 = [0 1.0 0 0.8;
    1.0 0 1.2 0;
    0 1.2 0 0.9;
    0.8 0 0.9 0];

% Tipos de nodos
% 1 = Lorenz, 2 = Rossler
nodeTypes = [1 1 2 2];

% Matriz Gamma
Gamma = diag([1 0.5 0.2]);

% Acoplamiento no lineal
nonlinear_coupling = true;

% Pesos variables en el tiempo
time_varying_weights = true;

% Condicion inicial total
X0 = zeros(N*n,1);
for i = 1:N
    X0((i-1)*n+1:i*n) = randn(n,1);
end