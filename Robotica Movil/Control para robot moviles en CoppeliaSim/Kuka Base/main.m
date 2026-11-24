clear all
close all
clc

bot = Bot_youBot_Platform();

L = 0.471/2;
l = 0.3/2;

dt = 0.05;
S = 10;

p = bot.Get_Pose();

p_plot = [];
t_plot = [];

for t=dt:dt:S
    v = [0.1 0.1 0.1 0.1]';
    

    
    bot.Set_Joint_Velocity(v);
    bot.Simulation_Step();
    
    p = bot.Get_Pose();
    disp(p')

    t_plot = [t_plot t];
    p_plot = [p_plot p];
end

bot.Stop_Simulation();

%%
figure
hold on
grid on
plot(p_plot(1,:),p_plot(2,:),'b-','LineWidth',2)
Dibujar_Omnidireccional_4(p,L,l)
