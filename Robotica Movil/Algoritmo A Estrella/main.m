clear all
close all
clc

img_color = imread('map_4.jpg');
img_resized = imresize(img_color,0.1);

% un click para inicio (x(1),y(1)), otro click para meta (x(2),y(2))
imshow(img_resized)
[x,y] = ginput(2);
x = round(x); y = round(y);
close

map = Mapa_Rejillas_Ocupacion(img_resized,[0 255 0],180);

%%
start = [y(1); x(1)];
goal = [y(2); x(2)];

astar = AStarBase(map);
path = astar.find(start,goal);

figure
imshow(map)
hold on
plot(path(2,:),path(1,:),'g-','LineWidth',2)
plot(start(2),start(1),'r.','LineWidth',2,'MarkerSize',15)
plot(goal(2),goal(1),'g.','LineWidth',2,'MarkerSize',15)

%%
map_dilate = imdilate(map,strel('square',2));

figure
imshow(map_dilate)



%%
function map = Mapa_Rejillas_Ocupacion (img,color,r)
    img = double(img);
    img_binary = (img(:,:,1)-color(1)).^2 + (img(:,:,2)-color(2)).^2 + (img(:,:,3)-color(3)).^2 < r^2;
    map = double(img_binary);
end