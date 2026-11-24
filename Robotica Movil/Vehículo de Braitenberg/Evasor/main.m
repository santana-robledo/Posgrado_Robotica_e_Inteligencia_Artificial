clear all
close all
clc

bot = DR12();
bot.Set_Joint_Velocity(0,0);

vn = 0;


d_max = 1;
d_min = 0.1;

dt = 0.01;
t_old = 0;  % dt = t - t_old
tic;

while toc<=60
    t = toc;

    if t-t_old >= dt
        t_old = t_old + dt;

        d = bot.Read_Sensors();




        vl = vn;
        vr = vn;

        bot.Set_Joint_Velocity(vl,vr);

        disp('Sensor readings:')
        disp(d)
    end
end

bot.Stop_Simulation();