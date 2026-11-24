clear all
close all
clc

bot = Bot_youBot();

dt = 0.05;
S = 5;

pb = bot.Get_Platform_Pose(); % x, y, theta
qa = bot.Get_Arm_Position(); % q1, q2, q3, q4, q5
xf = bot.Get_Reference();

q = [pb; qa]
xf


for t=dt:dt:S
    pb = bot.Get_Platform_Pose();
    qa = bot.Get_Arm_Position();
    




    
    qp = [0.3 0 0 0 0]'; % brazo
    v = [0.2 0.2 0.2 0.2]'; % plataforma

    bot.Set_Joint_Velocity(qp);
    bot.Set_Platform_Velocity(v);
    bot.Simulation_Step();
end

bot.Stop_Simulation();
