function A = build_topology(N, topology, params)

A = zeros(N);

switch lower(topology)
    
    case 'ring'
        % red de anillo: k vecinos por lado
        k = params.k;
        for i = 1:N
            for d = 1:k
                j1 = mod(i-1+d, N) + 1;
                j2 = mod(i-1-d, N) + 1;
                A(i,j1) = 1;
                A(i,j2) = 1;
            end
        end
        
    case 'regular'
        % red regular tipo anillo
        k = params.k;
        for i = 1:N
            for d = 1:k
                j1 = mod(i-1+d, N) + 1;
                j2 = mod(i-1-d, N) + 1;
                A(i,j1) = 1;
                A(i,j2) = 1;
            end
        end
        
    case 'smallworld'
        % modelo simplificado Watts-Strogatz
        k = params.k;
        p = params.p;
        
        % anillo inicial
        for i = 1:N
            for d = 1:k
                j = mod(i-1+d, N) + 1;
                A(i,j) = 1;
                A(j,i) = 1;
            end
        end
        
        % reconexion
        for i = 1:N
            for j = i+1:N
                if A(i,j) == 1 && rand < p
                    A(i,j) = 0; 
                    A(j,i) = 0;
                    
                    newj = randi(N);
                    while newj == i || A(i,newj) == 1
                        newj = randi(N);
                    end
                    
                    A(i,newj) = 1;
                    A(newj,i) = 1;
                end
            end
        end
        
    case 'random'
        pr = params.pr;
        for i = 1:N
            for j = i+1:N
                if rand < pr
                    A(i,j) = 1;
                    A(j,i) = 1;
                end
            end
        end
        
    case 'exponential'
        lambda = params.lambda;
        for i = 1:N
            for j = 1:N
                if i ~= j
                    d = min(abs(i-j), N-abs(i-j));
                    val = exp(-lambda*d);
                    if val > 0.2
                        A(i,j) = 1;
                    end
                end
            end
        end
        
    otherwise
        error('Topologia no reconocida');
end

% Asegurar simetria
A = double(A ~= 0);
A = max(A, A');
A(1:N+1:end) = 0;
end