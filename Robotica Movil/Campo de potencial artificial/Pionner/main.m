clear all
close all
clc

bot = Bot_Pioneer();

R = 0.195/2;
L = 0.381;
D = 0.08;

S = 60;
dt = 0.05;

for t=dt:dt:S
    p = bot.Get_Pose();
    disp('Pose:')
    disp(p)
    
    pg = bot.Get_Goal_Position();
    disp('Meta:')
    disp(pg)

    po = bot.Get_Obs_Positions();
    disp('Objetos:')
    disp(po)





    
    wr = 0;
    wl = 0;

    bot.Set_Joint_Velocity([wr wl]');  
    bot.Simulation_Step();
end

bot.Set_Joint_Velocity([0 0]');
bot.Stop_Simulation();
