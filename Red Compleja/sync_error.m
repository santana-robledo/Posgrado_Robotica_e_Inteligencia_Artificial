function E = sync_error(X, N, n) %Matriz completa de simulacion

Nt = size(X,1);
E = zeros(Nt,1);

for k = 1:Nt
    Xk = reshape(X(k,:), [n, N]); %Convertimos vector en matriz
    xmean = mean(Xk,2); %Promedio de los nodos
    
    e = 0;
    for i = 1:N
        e = e + norm(Xk(:,i) - xmean, 2);
    end
    E(k) = e/N;
end
end