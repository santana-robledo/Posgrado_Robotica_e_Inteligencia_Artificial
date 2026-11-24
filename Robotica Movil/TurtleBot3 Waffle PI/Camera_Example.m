clear all
close all
clc

bot = TurtleBot3_WafflePI('http://192.168.50.105:11311');

S = 60;

tic;
while toc<=S
    img = bot.Get_Image();
 
    cla
    imshow(img)
    drawnow
end
