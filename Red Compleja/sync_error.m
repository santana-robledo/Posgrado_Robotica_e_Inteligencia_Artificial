function E = sync_error(X, N, n)

Nt = size(X,1);
E = zeros(Nt,1);

for k = 1:Nt
    Xk = reshape(X(k,:), [n, N]);
    xmean = mean(Xk,2);
    
    e = 0;
    for i = 1:N
        e = e + norm(Xk(:,i) - xmean, 2);
    end
    E(k) = e/N;
end
end