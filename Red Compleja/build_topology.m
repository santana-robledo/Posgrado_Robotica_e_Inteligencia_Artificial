function A = build_topology(N, topology, params)

A = zeros(N);

switch lower(topology)
    
    case 'ring'
        % red de anillo: k vecinos por lado
        k = params.k;
        for i = 1:N
            for d = 1:k
                j1 = mod(i-1+d, N) + 1; %Vecino derecho
                j2 = mod(i-1-d, N) + 1; %Vecino izquierdo
                A(i,j1) = 1; %Crear conexiones
                A(i,j2) = 1;
            end
        end
        
    case 'regular'
    
        k = params.k * 2; % grado total deseado
        
        if mod(N*k,2) ~= 0
            error('N*k debe ser par');
        end
    
        grados = zeros(N,1);
    
        while any(grados < k)
    
            i = randi(N);
            j = randi(N);
    
            if i ~= j && A(i,j)==0
    
                if grados(i) < k && grados(j) < k
    
                    A(i,j)=1;
                    A(j,i)=1;
    
                    grados(i)=grados(i)+1;
                    grados(j)=grados(j)+1;
    
                end
            end
        end
        
    case 'smallworld'
        % modelo simplificado Watts-Strogatz
        k = params.k; %Numero vecinos
        p = params.p; %Probabilidad de reconexion
        
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
                if A(i,j) == 1 && rand < p %Existe conexion y despues aletoriedad
                    A(i,j) = 0; %Borrar conexión
                    A(j,i) = 0;
                    
                    newj = randi(N);%Crear nuevo
                    while newj == i || A(i,newj) == 1 %Evita que la reconexion se repita
                        newj = randi(N);
                    end
                    %Se crea nuevo enlace aleatorio
                    A(i,newj) = 1;
                    A(newj,i) = 1;
                end
            end
        end
        
    case 'random'
        pr = params.pr;
        for i = 1:N
            for j = i+1:N
                if rand < pr %Si menor a probabilidad se crea nodo
                    A(i,j) = 1;
                    A(j,i) = 1;
                end
            end
        end
        
    case 'exponential'
        lambda = params.lambda; %Velocidad de decaimiento
        for i = 1:N
            for j = 1:N
                if i ~= j
                    d = min(abs(i-j), N-abs(i-j)); %Calcula distancia mas corta
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
A(1:N+1:end) = 0;%Elimina diagonal principal
end