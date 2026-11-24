








clear all
close all
clc


bot = Bot_youBot();


return
dt = 0.05;
S = 10;

qp_plot = [];
t_plot = [];

for t=dt:dt:S
    Td = bot.ReferenceFrame_Pose();
    disp('Pose de referencia:')
    Td

    q = bot.Get_Joint_Position();
    disp('Valores articulares actuales:')
    q'

    qp = [0.1 0.0 0 0 0]';

    
    bot.Set_Joint_Velocity(qp);
    bot.Simulation_Step();

    qp_plot = [qp_plot qp];
    t_plot = [t_plot t];
end

bot.Stop_Simulation();


figure
hold on
grid on
plot(t_plot,qp_plot','LineWidth',2)
legend('\theta_1','\theta_2','\theta_3','\theta_4','\theta_5')
xlabel('segundos')
ylabel('rad/s')
