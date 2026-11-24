clear all
close all
clc

L = 0.471/2;
l = 0.3/2;

bot = Bot_youBot_Platform(0); % 0: no sincronizar, 1: sincronizar
bot.Set_Joint_Velocity([0 0 0 0]');

S = 60;

focalLength = [589.37 589.37]; 
principalPoint = [320 240];
imageSize = [480 640];
K = cameraIntrinsics(focalLength,principalPoint,imageSize);

tagSize = 0.3;
Xi = [-tagSize/2 tagSize/2 0 1; tagSize/2 tagSize/2 0 1; tagSize/2 -tagSize/2 0 1; -tagSize/2 -tagSize/2 0 1]';




tic;
while toc<=S
    I = bot.Get_Image();
    [~,uv,Rt] = readAprilTag(I,"tag36h11",K,tagSize);

    if ~isempty(uv) 
        % To get Z
        cTo = Rt.A;
        cXi = cTo*Xi;
        Z = cXi(3,:);






  
        v = [0 0 0 0]';
 
        bot.Set_Joint_Velocity(v);
    


        % Just drawings
        I = insertShape(I,"FilledCircle",[uv(1,:) 5],"Color","red","Opacity",1);
        I = insertShape(I,"FilledCircle",[uv(2,:) 5],"Color","green","Opacity",1);
        I = insertShape(I,"FilledCircle",[uv(3,:) 5],"Color","blue","Opacity",1);
        I = insertShape(I,"FilledCircle",[uv(4,:) 5],"Color","cyan","Opacity",1);

        I = insertText(I,uv(1,:),Z(1),BoxOpacity=1,FontSize=15);
        I = insertText(I,uv(2,:),Z(2),BoxOpacity=1,FontSize=15);
        I = insertText(I,uv(3,:),Z(3),BoxOpacity=1,FontSize=15);
        I = insertText(I,uv(4,:),Z(4),BoxOpacity=1,FontSize=15);

        imagePoints = worldToImage(K,Rt,[0 0 0; tagSize/2 0 0; 0 tagSize/2 0; 0 0 tagSize/2]);
        I = insertShape(I,Line=[imagePoints(1,:) imagePoints(2,:); ...
                        imagePoints(1,:) imagePoints(3,:); imagePoints(1,:) imagePoints(4,:)], ...
                        Color=["red","green","blue"],LineWidth=7);
    end

    cla
    imshow(I)
    drawnow

    bot.Simulation_Step(); % No tiene efecto en modo "no sincronizar"
end

bot.Stop_Simulation();



figure
hold on
grid on
plot(t_plot,e_plot,'LineWidth',2)
title('Errores e_i')

figure
hold on
grid on
plot(t_plot,sd_plot,'LineWidth',2)
plot(t_plot,s_plot,'LineWidth',2)
title('Deseados sd vs actuales s')




function s = Obtener_Punto_2D (uv,K)
        u = uv(1);
        v = uv(2);

        u0 = K(1,3);
        v0 = K(2,3);
        fu = K(1,1);
        fv = K(2,2);

        x = (u-u0)/fu;
        y = (v-v0)/fv;

        s = [x y]';
end

function Ls = Matriz_Interaccion (s,Z)
        x = s(1);
        y = s(2);

        Ls = [-1/Z 0 x/Z x*y -(1+x^2) y; 
              0 -1/Z y/Z 1+y^2 -x*y -x];
end