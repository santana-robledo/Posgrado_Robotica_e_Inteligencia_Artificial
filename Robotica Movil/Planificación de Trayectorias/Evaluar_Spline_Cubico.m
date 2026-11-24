function [qk,qpk,qppk] = Evaluar_Spline_Cubico (t,S,t_current)
    n = numel(t);
    
    if t_current>t(n)
        k = n-1;
    else
        k = find(t(1:n-1)<=t_current & t_current<=t(2:n),1);
    end
    
    qk = S(k,:)*[(t_current-t(k))^3 (t_current-t(k))^2 (t_current-t(k)) 1]';
    qpk = S(k,:)*[3*(t_current-t(k))^2 2*(t_current-t(k)) 1 0]';
    qppk = S(k,:)*[6*(t_current-t(k)) 2 0 0]';
end