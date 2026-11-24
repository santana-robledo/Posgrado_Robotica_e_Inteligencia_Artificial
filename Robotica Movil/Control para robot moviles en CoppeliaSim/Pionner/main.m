clear all
close all
clc



bot = Bot_Pioneer();

R = 0.195/2;
L = 0.381;



dt = 0.05;
S = tf;

p = bot.Get_Pose();

q_plot = [];
t_plot = [];
p_plot = [];


for t=dt:dt:S
    p = bot.Get_Pose();

    wr = 0.9;
    wl = 0.9;


    bot.Set_Joint_Velocity([wr wl]');
    bot.Simulation_Step();

    t_plot = [t_plot t];
    p_plot = [p_plot p];

end

bot.Stop_Simulation();


figure
plot(t_plot,p_plot)
