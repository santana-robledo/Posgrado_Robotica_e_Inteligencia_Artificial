clear all
close all
clc

S = 20;
dt = 0.01;

bot = Bot_Antro3DOF(dt);
Pd = bot.Get_References();

qf = IK(Pd(:,4));

qi = bot.Get_Position();

Qd = [qi IK(Pd(:,1)) IK(Pd(:,2)) IK(Pd(:,3)) IK(Pd(:,4)) qi];

td = linspace(0,S,6);

S1 = Coeficientes_Spline_Cubico(td,Qd(1,:));
S2 = Coeficientes_Spline_Cubico(td,Qd(2,:));
S3 = Coeficientes_Spline_Cubico(td,Qd(3,:));

M = @(q) Matriz_Inercia(q);
C = @(q,qp) Matriz_Coriolis(q,qp);
G = @(q) Vector_Gravedad(q);

% kp = 8*diag([2 2 1]);
% kd = 5*diag([2 2 1]);

kp = 0.1*diag([2 2 1]);
kd = 0.01*diag([2 2 1]);

x_plot = [];
u_plot = [];
t_plot = [];
qd_plot = [];

for t=dt:dt:S
    q = bot.Get_Position();
    qp = bot.Get_Velocity();
    
    [q1,qp1,qpp1] = Evaluar_Spline_Cubico(td,S1,t);
    [q2,qp2,qpp2] = Evaluar_Spline_Cubico(td,S2,t);
    [q3,qp3,qpp3] = Evaluar_Spline_Cubico(td,S3,t);

    qd = [q1 q2 q3]';
    qpd = [qp1 qp2 qp3]';
    qppd = [qpp1 qpp2 qpp3]';

    e = qd-q;
    ed = qpd-qp;

    % u = M(q)*(qppd + kp*e + kd*ed) + C(q,qp)*qp + G(q);
    u =  kp*e + kd*ed + M(q)*qppd + C(q,qp)*qpd + G(q);

    x = [q; qp];


    bot.Set_Force(u);
    bot.Simulation_Step();

    x_plot = [x_plot x];
    u_plot = [u_plot u];
    t_plot = [t_plot t];
    qd_plot = [qd_plot qd];
end

bot.Stop_Simulation();

figure
hold on
grid on
plot(t_plot,x_plot(1:3,:))
plot(t_plot,qd_plot)

figure
hold on
grid on
plot(t_plot,u_plot)


function qd = IK (xd)
    d_1 = 0.35;
    a_2 = 0.25;
    a_3 = 0.20;
    
    tx = xd(1);
    ty = xd(2);
    tz = xd(3);
    
    theta_3 = -acos((tx^2+ty^2+(tz-d_1)^2-a_2^2-a_3^2)/(2*a_2*a_3));
    theta_2 = atan2((tz-d_1),sqrt(tx^2+ty^2)) - asin((a_3*sin(theta_3))/(sqrt(tx^2+ty^2+(tz-d_1)^2)));
    theta_1 = atan2(ty,tx);
    
    qd = [theta_1 theta_2 theta_3]';
end