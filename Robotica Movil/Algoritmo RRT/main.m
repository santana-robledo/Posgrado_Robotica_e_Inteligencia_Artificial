clear all
close all
clc

xl = [-10 -10]';
xu = [10 10]';

xo = [2.5 2.5; -2.5 -2.5; 5.0 0.0]';
ro = [1.0 2.0 1.0];

D = 0.5;

x0 = [0 0]';
xd = [5.0 5.0]';


%% Algoritmo RRT
M = x0; % Coleccion de nodos
s = []; % Lista de nodos padres
t = []; % Lista de nodos hijos

iter = 1;

while iter<100000
    % Punto aleatorio xr
    xr = xl + (xu-xl).*rand(2,1);

    % Distancias entre xr y cada nodo en M 
    [d,i] = pdist2(M',xr','euclidean','Smallest',1);

    % Seleccion del nodo i mas cercano a xr
    xc = M(:,i);

    % Calcular xn
    v = (xr-xc)/norm(xr-xc);
    xn = xc + D*v;
    
    % Distancias entre xn y cada obstaculo en xo
    do = pdist2(xo',xn','euclidean');
    colisiones = (do<=ro')*1.0;

    % Si no hay colisiones
    if sum(colisiones)==0 && norm(xr-xc)>=D
        M = [M xn]; % Actualizamos coleccion de nodos
        s = [s i-1]; % Actualizamos nodos padres
        t = [t iter]; % Actualizamos nodos hijos

        iter = iter + 1;

        % Si estamos cerca del final xd
        if norm(xd-xn)<D
            M = [M xd];

            s = [s iter-1];
            t = [t iter];
            break
        end
    end
end

% [~,i] = pdist2(M',xd','euclidean','Smallest',1);
% 
% M = [M xd]; % Actualizamos coleccion de nodos
% s = [s i-1]; % Actualizamos nodos padres
% t = [t iter]; % Actualizamos nodos hijos

% Para obtener la ruta (path)
nodo_actual = t(end);
path = M(:,nodo_actual+1);

while nodo_actual~=0
    nodo_actual = s(t==nodo_actual);
    path = [M(:,nodo_actual+1) path];
end

%% Resultados
figure
hold on
grid on

plot(M(1,1),M(2,1),'bo','MarkerSize',10)

for i=1:numel(t)
    xc = M(:,s(i)+1);
    xn = M(:,t(i)+1);

    plot(xn(1),xn(2),'bo','MarkerSize',10)
    line([xc(1) xn(1)],[xc(2) xn(2)],'Color','b');
end

plot(x0(1),x0(2),'rx','MarkerSize',10,'LineWidth',2)
plot(xd(1),xd(2),'gx','MarkerSize',10,'LineWidth',2)

theta = linspace(-pi,pi,50);
for i=1:size(xo,2)
    plot(xo(1,i)+ro(i)*cos(theta),xo(2,i)+ro(i)*sin(theta),'r-','LineWidth',2)
end

plot(path(1,:),path(2,:),'g-','LineWidth',3)

axis equal

save('path.mat','path')