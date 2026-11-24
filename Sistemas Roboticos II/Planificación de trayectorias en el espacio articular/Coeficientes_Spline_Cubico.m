function S = Coeficientes_Spline_Cubico (t,q)
    [N,M] = size(q);

    if M>N
        q = q';
        t = t';
    end

    % Spline cúbico natural
    n = numel(t);
    
    h = diff(t); % t_{k+1} - t_{k}
    
    diag_sup_inf = h(2:n-2); % h_1, h_2, .... , h_{n-2}
    diag_princ = 2*(h(1:n-2)+h(2:n-1)); % 2*(h_0 + h_1), 2*(h_1 + h_2), ..., 2*(h_{n-2} + h_{n-1})
    
    M = diag(diag_princ) + diag(diag_sup_inf,1) + diag(diag_sup_inf,-1);
    
    f = diff(q)./diff(t); % f[t_k,t_{k+1}] = (q_{k+1} - q_{k})/(t_{k+1} - t_{k})
    b = 6*diff(f); % f[t_1,t_2]-f[t_0,t_1], f[t_2,t_3]-f[t_1,t_2], ..., f[t_{n-1},t_n]-f[t_{n-2},t_{n-1}]
    
    g = M\b; % g_1, g_2, ..., g_{n-1}
    
    g0 = 0; % frontera natural
    gn = 0; % frontera natural
    
    g = [g0; g; gn];
    
    % k = 0,1,2,...,n-1
    a = diff(g)./(6*h); % (g_{k+1}-g_k)/6*h_k
    b = g(1:n-1)/2; % g_k/2
    c = f - h.*(2*g(1:n-1)+g(2:n))/6; % f[t_k,t_{k+1}] - (2*h_k*g_k + g_{k+1}*h_k)/6
    d = q(1:n-1); % q_k
    
    S = [a b c d]; % coeficientes
end