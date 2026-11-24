clear all
close all
clc

S = 5;
dt = 0.01;

bot = Bot_Antro3DOF(dt);

xd = bot.Get_Reference();
disp(xd')

for t=dt:dt:S
    q = bot.Get_Position();
    qp = bot.Get_Velocity();
    
    u = [0 0 0]';
    bot.Set_Force(u);

    bot.Simulation_Step();
end

bot.Stop_Simulation();
