clear all
close all
clc

bot = Bot_dr12();

bot.Set_Joint_Velocity(0,0);
bot.Simulation_Step();

%%
img = bot.Get_Image();

[id,loc] = readAprilTag(img,"tag36h11");

pc = mean(loc(:,:,1));
pf = mean(loc(:,:,2));

%%
img_resized = imresize(img,0.2);
pc = round(pc*0.2);
pf = round(pf*0.2);

map = Mapa_Rejillas_Ocupacion(img_resized,[0 255 0],150);
map_dilate = imdilate(map,strel('square',15));

figure
imshow(map_dilate)


%%
S = 60;
dt = 0.05;

for t=dt:dt:S
    img = bot.Get_Image();
    [id,loc] = readAprilTag(img,"tag36h11");
    
	if ~isempty(id)
        p1 = loc(2,:,1);
        p2 = loc(3,:,1);
        pc = mean(loc(:,:,1));
        
        % p(x,y)
        theta = atan2(p2(2)-p1(2),p2(1)-p1(1));
        x = pc(1);
        y = pc(2);








        v = 0;
        w = 0;

        bot.Set_Joint_Velocity(v,-w);
        bot.Simulation_Step();
    end
end

bot.Stop_Simulation();


%%
function map = Mapa_Rejillas_Ocupacion (img,color,r)
    img = double(img);
    img_binary = (img(:,:,1)-color(1)).^2 + (img(:,:,2)-color(2)).^2 + (img(:,:,3)-color(3)).^2 < r^2;
    map = double(img_binary);
end
