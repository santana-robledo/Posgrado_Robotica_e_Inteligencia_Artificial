clear all
close all
clc

R = 0.195/2;
L = 0.381;

bot = Bot_Pioneer();
bot.Set_Joint_Velocity([0 0]');

S = 60;

D = 0.08;
d_max = 1.5;

ph_plot = [];
obs_plot = [];

tic
while toc<=S
    pg = bot.Get_Goal_Position();

    p = bot.Get_Pose();
    ph = p(1:2) + D*[cos(p(3)); sin(p(3))];

    [Angles,Ranges] = bot.Get_LaserScans();
    I = Ranges>d_max;
    Ranges(I) = [];
    Angles(I) = [];







    bot.Set_Joint_Velocity([0 0]');



    Xo = p(1) + Ranges.*cos(Angles+p(3));
    Yo = p(2) + Ranges.*sin(Angles+p(3));

    obs_plot = [obs_plot [Xo; Yo]];
    ph_plot = [ph_plot ph];


    [X,Y] = pol2cart(Angles,Ranges);

    cla
    plot(X,Y,'*')
    axis([-1.5*d_max 1.5*d_max -1.5*d_max 1.5*d_max])
    drawnow
end

bot.Stop_Simulation();

figure
hold on
grid on
plot(obs_plot(1,:),obs_plot(2,:),'rx','LineWidth',2)
plot(ph_plot(1,:),ph_plot(2,:),'-b','LineWidth',2)
plot(pg(1),pg(2),'og','LineWidth',2)
legend('Obs','Act')
