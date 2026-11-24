clear all
close all
clc

v = 0.5;
w =-0.2;

p = [0.5; 0.5; 0]; % x, y, theta


S = 60;
dt = 0.05;

p_plot = [];




for t=0:dt:S
    dot_p = [cos(p(3)) 0; sin(p(3)) 0; 0 1]*[v; w];

    p = p + dot_p*dt;

    p_plot = [p_plot p];

    cla
    Dibujar_Robot('Monociclo',p)
    plot(p_plot(1,:),p_plot(2,:),'--r','LineWidth',2)
    axis tight
    drawnow
end



return







return




p = [0.5; -0.5; pi/4];

figure
Dibujar_Robot('Omnidireccional_3w',p,L=0.3)

figure
Dibujar_Robot('Omnidireccional_4w',p,L=0.3)

figure
Dibujar_Robot('Omnidireccional_4wm',p,L1=0.25,l2=0.20)

figure
Dibujar_Robot('Coche',p,alpha=pi/4,d=0.25)

figure
Dibujar_Robot('Triciclo',p,alpha=pi/4,d=0.25)

figure
Dibujar_Robot('Monociclo_Desacoplado',p,d=0.25)

figure
Dibujar_Robot('Diferencial',p,L=0.20,R=0.10)

figure
Dibujar_Robot('Monociclo',p)

figure
Dibujar_Robot('Monociclo',p,Size=0.3)


