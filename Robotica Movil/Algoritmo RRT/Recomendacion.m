clear all
close all
clc

load path

% spath = Suavisar_Ruta (path,alpha,beta)
spath = Suavisar_Ruta(path,0.1,0.6);

figure
hold on
grid on

plot(path(1,:),path(2,:),'go-','LineWidth',2,'MarkerSize',10)
plot(spath(1,:),spath(2,:),'bo-','LineWidth',2,'MarkerSize',10)

legend('original','suavizada')


%%
function ps = Suavisar_Ruta (p,alpha,beta)
    ps = p;
    N = 5;
    M = size(p,2);

    for j=1:N
        for i=2:(M-1)
            ps(:,i) = ps(:,i) + alpha*(p(:,i)-ps(:,i)) + beta*(ps(:,i+1)+ps(:,i-1)-2*ps(:,i));
        end
    end
end
