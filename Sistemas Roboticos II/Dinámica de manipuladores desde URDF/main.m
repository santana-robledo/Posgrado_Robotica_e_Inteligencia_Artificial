clear all
close all
clc

S = 5;
dt = 0.01;

bot = Bot_KUKA_iiwa(dt);

for t=dt:dt:S
    q = bot.Get_Joint_Position();
    qp = bot.Get_Joint_Velocity();
    
    q'
    qp'

    u = [0 0 0 0 0 0 0]';

    bot.Set_Force(u);
    bot.Simulation_Step();
end

bot.Stop_Simulation();

%%
load urdf_kuka_iiwa.mat

showdetails(robot)

figure;
show(robot,q,'Frames','off');
axis equal;

endEffector = 'feltPen';

q = [0 0 0 -pi/4 0 pi/4 0]';
T07 = getTransform(robot,q,endEffector);
T07

qp = 0.1*ones(7,1);

M = massMatrix(robot,q);
C = velocityProduct(robot,q,qp); % M(q,qp)*qp
G = gravityTorque(robot,q);



