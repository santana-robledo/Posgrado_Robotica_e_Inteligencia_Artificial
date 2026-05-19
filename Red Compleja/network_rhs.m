function dx = network_rhs(t, x, params)
    N = params.N;
    n = params.n;
    c = params.c;
    A = params.A;
    W0 = params.W0;
    Gamma = params.Gamma;
    nodeTypes = params.nodeTypes;
    
    % Extraer estado de la referencia (final del vector)
    x_ref = x(N*n + 1 : end);
    dx_ref = node_dynamics(x_ref, params.ref_type);
    
    dx = zeros(N*n + n, 1);
    
    % Pesos variables
    if params.time_varying_weights
        alpha = 0.3; omega = 1.0;
        W = W0 .* (1 + alpha*sin(omega*t));
    else
        W = W0;
    end
    W = (W + W')/2; W = W .* A;

    for i = 1:N
        xi = x((i-1)*n+1 : i*n);
        fi = node_dynamics(xi, nodeTypes(i));
        
        % Acoplamiento
        coupling_sum = zeros(n,1);
        Hi = coupling_function(xi, params.nonlinear_coupling);
        for j = 1:N
            if A(i,j) ~= 0
                xj = x((j-1)*n+1 : j*n);
                Hj = coupling_function(xj, params.nonlinear_coupling);
                coupling_sum = coupling_sum + W(i,j) * Gamma * (Hj - Hi);
            end
        end
        
        % Control hacia referencia
        ui = zeros(n,1);
        if params.control_system
            ui = -params.K_gain * (xi - x_ref);
        end
        
        dx((i-1)*n+1 : i*n) = fi + c*coupling_sum + ui;
    end
    dx(N*n + 1 : end) = dx_ref;
end