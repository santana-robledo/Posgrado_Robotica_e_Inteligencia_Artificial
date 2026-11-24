clear all
close all
clc

L = 0.471/2;
l = 0.3/2;

bot = Bot_youBot_Platform();
bot.Set_Joint_Velocity([0 0 0 0]');

S = 30;




tic;
while toc<=S
    I = bot.Get_Image();

    [id,loc] = readAprilTag(I,"tag36h11");

    if ~isempty(id) 
        p1 = loc(1,:)';
        p2 = loc(2,:)';
        p3 = loc(3,:)';
        p4 = loc(4,:)';








        

        v = [0 0 0 0]'; % Poner el controlador
        bot.Set_Joint_Velocity(v);

        % Just drawings
        I = insertShape(I,'FilledCircle',[p1' 5],'Color','red','Opacity',1);
        I = insertShape(I,'FilledCircle',[p2' 5],'Color','green','Opacity',1);
        I = insertShape(I,'FilledCircle',[p3' 5],'Color','blue','Opacity',1);
        I = insertShape(I,'FilledCircle',[p4' 5],'Color','cyan','Opacity',1);
    end

    cla
    imshow(I)
    drawnow
end


bot.Stop_Simulation();
