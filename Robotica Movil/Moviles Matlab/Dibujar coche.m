function Dibujar_Coche (p,d)
    x = p(1);
    y = p(2);
    theta = p(3);
    alpha = p(4);
    
    D = 0.5;
    
    axis([-2 2 -2 2])
    hold on
    grid on
    xlabel('x')
    ylabel('y')
    
    % Ruedas laterales
    Tob = [cos(theta) -sin(theta) x; sin(theta) cos(theta) y; 0 0 1];
    Tbl = [1 0 0; 0 1 d*D; 0 0 1];
    Tbr = [1 0 0; 0 1 -d*D; 0 0 1];
    Tol = Tob*Tbl;
    Tor = Tob*Tbr;
    L = d*0.3;
    l = d*0.2;
    
    p1 = Tol*[+L -l 1]';
    p2 = Tol*[-L -l 1]';
    p3 = Tol*[+L +l 1]';
    p4 = Tol*[-L +l 1]';
    line([p1(1) p2(1)],[p1(2) p2(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])
    line([p1(1) p3(1)],[p1(2) p3(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])
    line([p2(1) p4(1)],[p2(2) p4(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])
    line([p3(1) p4(1)],[p3(2) p4(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])

    p1 = Tor*[+L -l 1]';
    p2 = Tor*[-L -l 1]';
    p3 = Tor*[+L +l 1]';
    p4 = Tor*[-L +l 1]';
    line([p1(1) p2(1)],[p1(2) p2(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])
    line([p1(1) p3(1)],[p1(2) p3(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])
    line([p2(1) p4(1)],[p2(2) p4(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])
    line([p3(1) p4(1)],[p3(2) p4(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])
    
    % Rueda delantera
    Tbl = [cos(alpha) -sin(alpha) d; sin(alpha) cos(alpha) d*D; 0 0 1];
    Tbr = [cos(alpha) -sin(alpha) d; sin(alpha) cos(alpha) -d*D; 0 0 1];
    Tol = Tob*Tbl;
    Tor = Tob*Tbr;
    
    p1 = Tol*[+L -l 1]';
    p2 = Tol*[-L -l 1]';
    p3 = Tol*[+L +l 1]';
    p4 = Tol*[-L +l 1]';
    line([p1(1) p2(1)],[p1(2) p2(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])
    line([p1(1) p3(1)],[p1(2) p3(2)],'LineWidth',2,'MarkerSize',10,'color',[1 0 0])
    line([p2(1) p4(1)],[p2(2) p4(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])
    line([p3(1) p4(1)],[p3(2) p4(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])
    
    p1 = Tor*[+L -l 1]';
    p2 = Tor*[-L -l 1]';
    p3 = Tor*[+L +l 1]';
    p4 = Tor*[-L +l 1]';
    line([p1(1) p2(1)],[p1(2) p2(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])
    line([p1(1) p3(1)],[p1(2) p3(2)],'LineWidth',2,'MarkerSize',10,'color',[1 0 0])
    line([p2(1) p4(1)],[p2(2) p4(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])
    line([p3(1) p4(1)],[p3(2) p4(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])
    
    % Base
    L = d;
    l = d;
    
    p1 = Tob*[+L+d*D -l 1]';
    p2 = Tob*[-L+d*D -l 1]';
    p3 = Tob*[+L+d*D +l 1]';
    p4 = Tob*[-L+d*D +l 1]';
    line([p1(1) p2(1)],[p1(2) p2(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])
    line([p1(1) p3(1)],[p1(2) p3(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])
    line([p2(1) p4(1)],[p2(2) p4(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])
    line([p3(1) p4(1)],[p3(2) p4(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])
    




    
    
%     % Base
%     phi = linspace(0,2*pi,50);
%     cx = x + D*cos(phi); 
%     cy = y + D*sin(phi);
%     plot(cx,cy,'LineWidth',2,'MarkerSize',10)
%     
%     % Rueda delantera
%     Tob = [cos(theta) -sin(theta) x; sin(theta) cos(theta) y; 0 0 1];
%     Tbf = [cos(alpha) -sin(alpha) d; sin(alpha) cos(alpha) 0; 0 0 1];
%     Tor = Tob*Tbf;
%     
%     l = d*0.3;
%     L = d*0.4;
%     p1 = Tor*[+L -l 1]';
%     p2 = Tor*[-L -l 1]';
%     p3 = Tor*[+L +l 1]';
%     p4 = Tor*[-L +l 1]';
%     line([p1(1) p2(1)],[p1(2) p2(2)],'LineWidth',2,'MarkerSize',10)
%     line([p1(1) p3(1)],[p1(2) p3(2)],'LineWidth',2,'MarkerSize',10,'color',[1 0 0])
%     line([p2(1) p4(1)],[p2(2) p4(2)],'LineWidth',2,'MarkerSize',10)
%     line([p3(1) p4(1)],[p3(2) p4(2)],'LineWidth',2,'MarkerSize',10)