clear all
close all
clc

bot = TurtleBot3_WafflePI('http://192.168.50.105:11311');

%%
S = 10;

v = 0.2;
w = 0.0;

bot.Reset_Odometry();

tic;
while toc<=S
    t = toc;



    
    p = bot.Get_Pose();
    % bot.Set_Velocity(v,w);



    disp(p')
end

bot.Set_Velocity(0,0);


