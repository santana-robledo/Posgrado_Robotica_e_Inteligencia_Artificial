clear all
close all
clc

bot = TurtleBot3_WafflePI('http://192.168.50.105:11311');

S = 30;

tic;
while toc<=S
    [Angles,Ranges] = bot.Get_LaserScans();
    [X,Y] = pol2cart(Angles,Ranges);
    
    cla
    hold on
    grid on
    scatter(X,Y,'filled')
    xlabel('x')
    ylabel('y')
    axis([-3 3 -3 3])
    drawnow
end
