clear all
close all
clc

%%
img_color = imread('map_1.jpg');
img_resized = imresize(img_color,0.1);

% un click para inicio (x(1),y(1)), otro click para meta (x(2),y(2))
imshow(img_resized)
[x,y] = ginput(2);
x = round(x); y = round(y);
close

map = Mapa_Rejillas_Ocupacion(img_resized,[0 255 0],180);

imshow(map)


%%
start = [y(1); x(1)];
goal = [y(2); x(2)];

wf = Wavefront(map);
path = wf.Find(start,goal);

% bf = Breadth_First_Search(map);
% path = bf.Find(start,goal);



figure
imshow(map)
hold on
plot(path(2,:),path(1,:),'g-','LineWidth',2)
plot(start(2),start(1),'g.','LineWidth',2,'MarkerSize',15)
plot(goal(2),goal(1),'r.','LineWidth',2,'MarkerSize',15)

%%
function map = Mapa_Rejillas_Ocupacion (img,color,r)
    img = double(img);
    img_binary = (img(:,:,1)-color(1)).^2 + (img(:,:,2)-color(2)).^2 + (img(:,:,3)-color(3)).^2 < r^2;
    map = double(img_binary);
end