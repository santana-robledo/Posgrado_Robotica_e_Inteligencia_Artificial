function [Fx,Fy] = F_Atraccion (k_att,xg,yg,x,y)
    Fx = k_att*(xg-x);
    Fy = k_att*(yg-y);