clear all
close all
clc


bot = Bot_Pioneer();

dt = 0.05;
S = 5;

pb = bot.Get_Platform_Pose();
qa = bot.Get_Arm_Position();
xf = bot.Get_Reference();

q = [pb; qa]
xf

for t=dt:dt:S
    pb = bot.Get_Platform_Pose();
    qa = bot.Get_Arm_Position();



    wr = 0.6; % plataforma
    wl = 0.6; % plataforma

    qp = [-0.2 0 0]'; % brazo

    bot.Set_Platform_Velocity([wr wl]');
    bot.Set_Joint_Velocity(qp);
    bot.Simulation_Step();
end

bot.Stop_Simulation();

