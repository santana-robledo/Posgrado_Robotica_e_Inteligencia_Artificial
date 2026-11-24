%%
function Dibujar_Manipulador (A,Q,S)
    if nargin==2
        S = 0.05;
    end
    
    grid on
    hold on

    [~,n] = size(A);
    T = zeros(4,4,n);

    T(:,:,1) = A{1};
    
    Dibujar_Marco_Referencia(eye(4,4),S,0)

    if Q{1}(1)=='R'
        Dibujar_Rotacional(eye(4,4),S)
    else
        Dibujar_Prismatica(eye(4,4),S)
    end

    line([0 T(1,4,1)],[0 T(2,4,1)],[0 T(3,4,1)],'color',[0 0 0],'LineWidth',1,'LineStyle','--')

    for i=2:n
        T(:,:,i) = T(:,:,i-1)*A{i};

        if norm(T(1:3,4,i-1)-T(1:3,4,i))
            Dibujar_Marco_Referencia(T(:,:,i-1),S,i-1)
        else
            Dibujar_Marco_Referencia(T(:,:,i-1),2*S,i-1)
        end

        if Q{1}(i)=='R'
            Dibujar_Rotacional(T(:,:,i-1),S)
        else
            Dibujar_Prismatica(T(:,:,i-1),S)
        end

        line([T(1,4,i-1) T(1,4,i)],[T(2,4,i-1) T(2,4,i)],[T(3,4,i-1) T(3,4,i)],'color',[0 0 0],'LineWidth',1,'LineStyle','--')
    end

    Dibujar_Marco_Referencia(T(:,:,n),S,n)
 
    view([-60 30])
    axis equal
end

%%
function Dibujar_Rotacional (T,S)
    if nargin==1
        S = 0.05;
    end

    n = 50;
    r = S;
    
    alpha = linspace(-pi,pi,n);
    Xl = r*cos(alpha)';
    Yl = r*sin(alpha)';
    Zl = S*ones(n,1);
    
    Xu = Xl;
    Yu = Yl;
    Zu = -Zl;
    
    XYZ = [Xl Yl Zl ones(n,1)]*T';
    Xl = XYZ(:,1);
    Yl = XYZ(:,2);
    Zl = XYZ(:,3);
    
    XYZ = [Xu Yu Zu ones(n,1)]*T';
    Xu = XYZ(:,1);
    Yu = XYZ(:,2);
    Zu = XYZ(:,3);
    
    fill3(Xl,Yl,Zl,'blue','FaceAlpha',0.0,'LineWidth',1);
    fill3(Xu,Yu,Zu,'blue','FaceAlpha',0.0,'LineWidth',1);
    
    line([Xl(1) Xu(1)],[Yl(1) Yu(1)],[Zl(1) Zu(1)],'color',[0 0 0],'linewidth',1)
    line([Xl(13) Xu(13)],[Yl(13) Yu(13)],[Zl(13) Zu(13)],'color',[0 0 0],'linewidth',1)
    line([Xl(26) Xu(26)],[Yl(26) Yu(26)],[Zl(26) Zu(26)],'color',[0 0 0],'linewidth',1)
    line([Xl(38) Xu(38)],[Yl(38) Yu(38)],[Zl(38) Zu(38)],'color',[0 0 0],'linewidth',1)
end

%%
function Dibujar_Prismatica (T,S)
    if nargin==1
        S = 0.05;
    end

    X = S*[-1 -1 -1 -1 -1 1; 1 -1 1 1 1 1; 1 -1 1 1 1 1; -1 -1 -1 -1 -1 1];
    Y = S*[-1 -1 -1 -1 1 -1; -1 1 -1 -1 1 1; -1 1 1 1 1 1; -1 -1 1 1 1 -1];
    Z = S*[-1 -1 1 -1 -1 -1; -1 -1 1 -1 -1 -1; 1 1 1 -1 1 1; 1 1 1 -1 1 1];

    Yf = X(:,3);
    Xf = Y(:,3);
    Zf = Z(:,3) + S*0.2;

    XYZ = [reshape(X,24,1) reshape(Y,24,1) reshape(Z,24,1) ones(24,1)]*T';
    X = reshape(XYZ(:,1),4,6);
    Y = reshape(XYZ(:,2),4,6);
    Z = reshape(XYZ(:,3),4,6);

    XYZ = [Xf Yf Zf ones(4,1)]*T';
    Xf = XYZ(:,1);
    Yf = XYZ(:,2);
    Zf = XYZ(:,3);
    
    fill3(X,Y,Z,'blue','FaceAlpha',0.0,'LineWidth',1)
    fill3(Xf,Yf,Zf,'blue','FaceAlpha',0.0,'LineWidth',1)
end

%%
function Dibujar_Marco_Referencia (T,S,indx)
    if nargin==1
        S = 0.05;
    end

    px = T*[1.5*S 0 0 1]';
    py = T*[0 1.5*S 0 1]';
    pz = T*[0 0 1.5*S 1]';
    
    t = T(1:3,4);
    
    line([t(1) px(1)],[t(2) px(2)],[t(3) px(3)],'color',[1 0 0],'LineWidth',2)
    line([t(1) py(1)],[t(2) py(2)],[t(3) py(3)],'color',[0 1 0],'LineWidth',2)
    line([t(1) pz(1)],[t(2) pz(2)],[t(3) pz(3)],'color',[0 0 1],'LineWidth',2)

    indx = num2str(indx);
    text(px(1),px(2),px(3),['x_' indx],'Color','blue','FontSize',15)
    text(py(1),py(2),py(3),['y_' indx],'Color','blue','FontSize',15)
    text(pz(1),pz(2),pz(3),['z_' indx],'Color','blue','FontSize',15)
end