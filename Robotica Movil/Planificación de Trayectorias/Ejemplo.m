clear all
close all
clc

xf = [1.0000 0.9458 0.7891 0.5469 0.2455 -0.0826 -0.4017 -0.6773 -0.8795 ...
     -0.9864 -0.9864 -0.8795 -0.6773 -0.4017 -0.0826 0.2455 0.5469 0.7891 0.9458 1.0000];

S = 30;
n = numel(xf);

tf = linspace(0,S,n);
dt = 0.05;

%% Spline Lineal
xd_plot = [];
dot_xd_plot = [];
t_plot = [];

for t=0:dt:S
    [xd,dot_xd] = Evaluar_Spline_Lineal(tf,xf,t);

    xd_plot = [xd_plot xd];
    dot_xd_plot = [dot_xd_plot dot_xd];
    t_plot = [t_plot t];
end

figure
hold on
grid on
plot(tf,xf,'o','LineWidth',2)
xlabel('t [m]','FontSize',14)
ylabel('x [m]','FontSize',14)
set(gca, 'FontSize', 16)

figure

subplot(2,1,1)
hold on
grid on
plot(tf,xf,'o','LineWidth',2)
plot(t_plot,xd_plot,'LineWidth',2)
xlabel('t [s]','FontSize',14)
ylabel('x [m]','FontSize',14)
set(gca, 'FontSize', 16)

subplot(2,1,2)
hold on
grid on
plot(t_plot,dot_xd_plot,'LineWidth',2)
xlabel('t [s]','FontSize',14)
ylabel('x [m/s]','FontSize',14)
set(gca, 'FontSize', 16)

%% Spline Cubico
xd_plot = [];
dot_xd_plot = [];
ddot_xd_plot = [];
t_plot = [];

Sx = Coeficientes_Spline_Cubico(tf,xf);


for t=0:dt:S
    [xd,dot_xd,ddot_xd] = Evaluar_Spline_Cubico(tf,Sx,t);

    xd_plot = [xd_plot xd];
    dot_xd_plot = [dot_xd_plot dot_xd];
    ddot_xd_plot = [ddot_xd_plot ddot_xd];
    t_plot = [t_plot t];
end

figure
hold on
grid on
plot(tf,xf,'o','LineWidth',2)
xlabel('t [m]','FontSize',14)
ylabel('x [m]','FontSize',14)
set(gca, 'FontSize', 16)

figure

subplot(3,1,1)
hold on
grid on
plot(tf,xf,'o','LineWidth',2)
plot(t_plot,xd_plot,'LineWidth',2)
xlabel('t [s]','FontSize',14)
ylabel('x [m]','FontSize',14)
set(gca, 'FontSize', 16)

subplot(3,1,2)
hold on
grid on
plot(t_plot,dot_xd_plot,'LineWidth',2)
xlabel('t [s]','FontSize',14)
ylabel('x [m/s]','FontSize',14)
set(gca, 'FontSize', 16)

subplot(3,1,3)
hold on
grid on
plot(t_plot,ddot_xd_plot,'LineWidth',2)
xlabel('t [s]','FontSize',14)
ylabel('x [m/s^2]','FontSize',14)
set(gca, 'FontSize', 16)






return



figure
hold on
grid on
plot(xf,yf,'o','LineWidth',2)
plot(pd_plot(1,:),pd_plot(2,:),'LineWidth',2)
xlabel('x [m]','FontSize',14)
ylabel('y [m]','FontSize',14)
set(gca, 'FontSize', 16)

figure
title('Referencia en posicion')

subplot(2,1,1)
hold on
grid on

plot(tf,xf,'o','LineWidth',2)
legend('x_d','x_f','FontSize',14)
xlabel('t [s]','FontSize',14)
ylabel('x [m]','FontSize',14)
set(gca, 'FontSize', 16)

subplot(2,1,2)
hold on
grid on
plot(t_plot,pd_plot(2,:),'LineWidth',2)
plot(tf,yf,'o','LineWidth',2)
legend('y_d','y_f','FontSize',14)
xlabel('t [s]','FontSize',14)
ylabel('y [m]','FontSize',14)
set(gca, 'FontSize', 16)

figure
title('Referencia en velocidad')

subplot(2,1,1)
hold on
grid on
plot(t_plot,dot_pd_plot(1,:),'LineWidth',2)
legend('ẋ_d', 'FontSize', 16)
xlabel('t [s]')
ylabel('ẋ [m/s]')
set(gca, 'FontSize', 16)

subplot(2,1,2)
hold on
grid on
plot(t_plot,dot_pd_plot(2,:),'LineWidth',2)
legend('ẏ_d', 'FontSize', 16)
xlabel('t [s]')
ylabel('ẏ [m/s]')
set(gca, 'FontSize', 16)


%%
xf = [0.00 -0.15 -0.35 -0.55 -0.75 -0.95 -1.10 -1.20 -1.25 -1.25 -1.20 ...
     -1.10 -0.95 -0.75 -0.55 -0.35 -0.15 0.00 0.15 0.35 0.55 0.75 0.95 ...
     1.10 1.20 1.25 1.25 1.20 1.10 0.95 0.75 0.55 0.35 0.15 0.00];

yf = [-1.25 -1.05 -0.85 -0.65 -0.45 -0.25 -0.05 0.15 0.35 0.55 0.75 0.95 ...
     1.10 1.20 1.25 1.20 1.05 0.90 1.05 1.20 1.25 1.20 1.10 0.95 0.75 ...
     0.55 0.35 0.15 -0.05 -0.25 -0.45 -0.65 -0.85 -1.05 -1.25];

S = 60;
n = numel(xf);

tf = linspace(0,S,n);
dt = 0.05;

Sx = Obtener_Spline_Cubico(tf',xf');
Sy = Obtener_Spline_Cubico(tf',yf');

pf_plot = [];
pd_plot = [];
dot_pd_plot = [];
ddot_pd_plot = [];
t_plot = [];

for t=0:dt:S
    [xd,dot_xd,ddot_xd] = Evaluar_Spline_Cubico(tf,Sx,t);
    [yd,dot_yd,ddot_yd] = Evaluar_Spline_Cubico(tf,Sy,t);

    pd_plot = [pd_plot [xd; yd]];
    dot_pd_plot = [dot_pd_plot [dot_xd; dot_yd]];
    ddot_pd_plot = [ddot_pd_plot [ddot_xd; ddot_yd]];
    t_plot = [t_plot t];
end

figure
hold on
grid on
plot(xf,yf,'o','LineWidth',2)
xlabel('x [m]','FontSize',14)
ylabel('y [m]','FontSize',14)
set(gca, 'FontSize', 16)

figure
title('Referencia en posicion')

subplot(2,1,1)
hold on
grid on
plot(tf,xf,'o','LineWidth',2)
xlabel('t [s]','FontSize',14)
ylabel('x [m]','FontSize',14)
set(gca, 'FontSize', 16)

subplot(2,1,2)
hold on
grid on
plot(tf,yf,'o','LineWidth',2)
xlabel('t [s]','FontSize',14)
ylabel('y [m]','FontSize',14)
set(gca, 'FontSize', 16)

figure
hold on
grid on
plot(xf,yf,'o','LineWidth',2)
plot(pd_plot(1,:),pd_plot(2,:),'LineWidth',2)
xlabel('x [m]','FontSize',14)
ylabel('y [m]','FontSize',14)
set(gca, 'FontSize', 16)

figure
title('Referencia en posicion')

subplot(2,1,1)
hold on
grid on
plot(t_plot,pd_plot(1,:),'LineWidth',2)
plot(tf,xf,'o','LineWidth',2)
legend('x_d','x_f','FontSize',14)
xlabel('t [s]','FontSize',14)
ylabel('x [m]','FontSize',14)
set(gca, 'FontSize', 16)

subplot(2,1,2)
hold on
grid on
plot(t_plot,pd_plot(2,:),'LineWidth',2)
plot(tf,yf,'o','LineWidth',2)
legend('y_d','y_f','FontSize',14)
xlabel('t [s]','FontSize',14)
ylabel('y [m]','FontSize',14)
set(gca, 'FontSize', 16)

figure
title('Referencia en velocidad')

subplot(2,1,1)
hold on
grid on
plot(t_plot,dot_pd_plot(1,:),'LineWidth',2)
legend('ẋ_d', 'FontSize', 16)
xlabel('t [s]')
ylabel('ẋ [m/s]')
set(gca, 'FontSize', 16)

subplot(2,1,2)
hold on
grid on
plot(t_plot,dot_pd_plot(2,:),'LineWidth',2)
legend('ẏ_d', 'FontSize', 16)
xlabel('t [s]')
ylabel('ẏ [m/s]')
set(gca, 'FontSize', 16)


figure
title('Referencia en acceleracion')

subplot(2,1,1)
hold on
grid on
plot(t_plot,ddot_pd_plot(1,:),'LineWidth',2)
legend('ẍ_d', 'FontSize', 16)
xlabel('t [s]')
ylabel('ẍ [m/s^2]')
set(gca, 'FontSize', 16)

subplot(2,1,2)
hold on
grid on
plot(t_plot,ddot_pd_plot(2,:),'LineWidth',2)
legend('ÿ_d', 'FontSize', 16)
xlabel('t [s]')
ylabel('ÿ [m/s^2]')
set(gca, 'FontSize', 16)



%%




