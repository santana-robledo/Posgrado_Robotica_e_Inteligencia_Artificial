clear all
close all
clc

xg = 5.0;
yg = 5.0;

xo_1 = 2;
yo_1 = 4;

xo_2 = 3.01;
yo_2 = 3.0;

x = 0.0;
y = 0.0;

k_att = 0.3;
k_rep = 0.1;
phi = 1.5;

N = 500;
h = 0.1;

path = [x y]';

for i=1:N
    [Fa_x,Fa_y] = F_Atraccion(k_att,xg,yg,x,y);
    [Fr_x1,Fr_y1] = F_Repulsion(k_rep,phi,xo_1,yo_1,x,y);
    [Fr_x2,Fr_y2] = F_Repulsion(k_rep,phi,xo_2,yo_2,x,y);
    
    Fx = Fa_x + Fr_x1 + Fr_x2;
    Fy = Fa_y + Fr_y1 + Fr_y2;
    
    % x = x + h*Fx;
    % y = y + h*Fy;

    x = x + h*(Fx/sqrt(Fx^2+Fy^2));
    y = y + h*(Fy/sqrt(Fx^2+Fy^2));
    
    path = [path [x; y]];

    e = sqrt((xg-x)^2+(yg-y)^2);

    if e<0.1
        break
    end


 cla
hold on
grid on
xlabel('x')
ylabel('y')

plot(path(1,:),path(2,:),'b-','LineWidth',2)
plot(path(1,:),path(2,:),'bo','MarkerSize',10)

plot(xo_1,yo_1,'ro','LineWidth',2,'MarkerSize',10)
plot(xo_1,yo_1,'rx','LineWidth',2,'MarkerSize',10)

plot(xo_2,yo_2,'ro','LineWidth',2,'MarkerSize',10)
plot(xo_2,yo_2,'rx','LineWidth',2,'MarkerSize',10)    

plot(xg,yg,'go','LineWidth',2,'MarkerSize',10)
plot(xg,yg,'gx','LineWidth',2,'MarkerSize',10)

theta = linspace(-pi,pi,50);
plot(xo_1+phi*cos(theta),yo_1+phi*sin(theta),'r--')
plot(xo_2+phi*cos(theta),yo_2+phi*sin(theta),'r--')

drawnow
end


figure
hold on
grid on
xlabel('x')
ylabel('y')

plot(path(1,:),path(2,:),'b-','LineWidth',2)
plot(path(1,:),path(2,:),'bo','MarkerSize',10)

plot(xo_1,yo_1,'ro','LineWidth',2,'MarkerSize',10)
plot(xo_1,yo_1,'rx','LineWidth',2,'MarkerSize',10)

plot(xo_2,yo_2,'ro','LineWidth',2,'MarkerSize',10)
plot(xo_2,yo_2,'rx','LineWidth',2,'MarkerSize',10)    

plot(xg,yg,'go','LineWidth',2,'MarkerSize',10)
plot(xg,yg,'gx','LineWidth',2,'MarkerSize',10)

theta = linspace(-pi,pi,50);
plot(xo_1+phi*cos(theta),yo_1+phi*sin(theta),'r--')
plot(xo_2+phi*cos(theta),yo_2+phi*sin(theta),'r--')
