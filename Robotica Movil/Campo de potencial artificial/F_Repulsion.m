function [Fx,Fy] = F_Repulsion (k_rep,phi,xo,yo,x,y)
    phi_o = sqrt((x-xo)^2+(y-yo)^2);
    
    if phi_o<=phi
        Fx = k_rep*((1/phi_o)-(1/phi))*(1/phi_o^2)*((x-xo)/phi_o);
        Fy = k_rep*((1/phi_o)-(1/phi))*(1/phi_o^2)*((y-yo)/phi_o);
    else
        Fx = 0;
        Fy = 0;
    end