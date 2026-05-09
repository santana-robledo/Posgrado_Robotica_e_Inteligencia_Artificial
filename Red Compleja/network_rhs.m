function dx = network_rhs(t, x, params)

N = params.N;
n = params.n;
c = params.c;
A = params.A;
W0 = params.W0;
Gamma = params.Gamma;
nodeTypes = params.nodeTypes;

dx = zeros(N*n,1);

% Pesos variables en el tiempo
if params.time_varying_weights
    alpha = 0.3;
    omega = 1.0;
    W = W0 .* (1 + alpha*sin(omega*t));
else
    W = W0;
end

% Mantener simetria por seguridad numerica
W = (W + W')/2;
W = W .* A;

for i = 1:N
    xi = x((i-1)*n+1:i*n);
    
    % Dinamica interna del nodo
    fi = node_dynamics(xi, nodeTypes(i));
    
    % Acoplamiento
    coupling_sum = zeros(n,1);
    Hi = coupling_function(xi, params.nonlinear_coupling);
    
    for j = 1:N
        if A(i,j) ~= 0
            xj = x((j-1)*n+1:j*n);
            Hj = coupling_function(xj, params.nonlinear_coupling);
            
            coupling_sum = coupling_sum + W(i,j) * Gamma * (Hj - Hi);
        end
    end
    
    % Sin control
    ui = zeros(n,1);
    
    dx((i-1)*n+1:i*n) = fi + c*coupling_sum + ui;
end
end