function [Sk,dot_Sk] = Evaluar_Spline_Lineal (t,q,t_current)
    n = numel(t);
    
    if t_current>t(n)
        k = n-1;
    else
        k = find(t(1:n-1)<=t_current & t_current<=t(2:n),1);
    end
    
    Sk = q(k) + ((q(k+1)-q(k))/(t(k+1)-t(k)))*(t_current-t(k));
    dot_Sk = ((q(k+1)-q(k))/(t(k+1)-t(k)));
end