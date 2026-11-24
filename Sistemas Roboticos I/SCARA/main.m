clear all
close all
clc

bot = Bot_Scara();

%%
bot.Gripper_Command(0) % 0 para cerrar
pause(1)

bot.Gripper_Command(1) % 1 para abrir
pause(1)


%%
Td = bot.Rectangle_Pose();
Td

Td = bot.ReferenceFrame_Pose();
Td

%%
q = [pi/4 -pi/4 0 0]';
bot.Set_Joint_Position(q);
q = bot.Get_Joint_Position();
q'

pause(1)

q = [-pi/4 pi/4 0 0]';
bot.Set_Joint_Position(q);
q = bot.Get_Joint_Position();
q'

pause(1)

%%
S = 5;

tic
while toc<=S
    t = toc;
    disp(['time=' num2str(t)])



    
end
pause(1)

q = [0 0 0 0]';
bot.Set_Joint_Position(q);
pause(2)

%%
bot.Stop_Simulation();
