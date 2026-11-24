function Dibujar_Bicicleta (p,alpha,d)
    x = p(1);
    y = p(2);
    theta = p(3);
    
    axis([-2 2 -2 2])
    hold on
    grid on
    xlabel('x')
    ylabel('y')
    
    % Base
    cx = x + d*cos(theta); 
    cy = y + d*sin(theta);
    line([x cx],[y cy],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])
    
    % Rueda delantera
    Tob = [cos(theta) -sin(theta) x; sin(theta) cos(theta) y; 0 0 1];
    Tbf = [cos(alpha) -sin(alpha) d; sin(alpha) cos(alpha) 0; 0 0 1];
    Tor = Tob*Tbf;
    
    l = d*0.2;
    L = d*0.3;
    p1 = Tor*[+L -l 1]';
    p2 = Tor*[-L -l 1]';
    p3 = Tor*[+L +l 1]';
    p4 = Tor*[-L +l 1]';
    line([p1(1) p2(1)],[p1(2) p2(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])
    line([p1(1) p3(1)],[p1(2) p3(2)],'LineWidth',2,'MarkerSize',10,'color',[1 0 0])
    line([p2(1) p4(1)],[p2(2) p4(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])
    line([p3(1) p4(1)],[p3(2) p4(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])

    % Rueda trasera
    l = d*0.2;
    L = d*0.3;
    Tol = Tob;
    
    p1 = Tol*[+L -l 1]';
    p2 = Tol*[-L -l 1]';
    p3 = Tol*[+L +l 1]';
    p4 = Tol*[-L +l 1]';
    line([p1(1) p2(1)],[p1(2) p2(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])
    line([p1(1) p3(1)],[p1(2) p3(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])
    line([p2(1) p4(1)],[p2(2) p4(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])
    line([p3(1) p4(1)],[p3(2) p4(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])