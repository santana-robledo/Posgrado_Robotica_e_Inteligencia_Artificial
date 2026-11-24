function Dibujar_Movil (p,d)
	x = p(1);
    y = p(2);
    theta = p(3);

    L = 1.5*d;
    l = d*0.4;
    
    axis([-2 2 -2 2])
    hold on
    grid on
    xlabel('x')
    ylabel('y')
    
    % Base
    phi = linspace(0,2*pi,50);
    cx = x + L*cos(phi); 
    cy = y + L*sin(phi);
    plot(cx,cy,'b','LineWidth',2,'MarkerSize',10)
    
    % Marcador delantero
    Tob = [cos(theta) -sin(theta) x; sin(theta) cos(theta) y; 0 0 1];
    Tbf = [1 0 d; 0 1 0; 0 0 1];
    Tor = Tob*Tbf;
    
    cx = Tor(1,3) + (L*0.2)*cos(phi);
    cy = Tor(2,3) + (L*0.2)*sin(phi);
    plot(cx,cy,'r','LineWidth',2,'MarkerSize',10)
    
    % Ruedas laterales
    Tbl = [1 0 0; 0 1 L+l; 0 0 1];
    Tbr = [1 0 0; 0 1 -L-l; 0 0 1];
    Tol = Tob*Tbl;
    Tor = Tob*Tbr;
    
    p1 = Tol*[+L*0.5 -l 1]';
    p2 = Tol*[-L*0.5 -l 1]';
    p3 = Tol*[+L*0.5 +l 1]';
    p4 = Tol*[-L*0.5 +l 1]';
    line([p1(1) p2(1)],[p1(2) p2(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])
    line([p1(1) p3(1)],[p1(2) p3(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])
    line([p2(1) p4(1)],[p2(2) p4(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])
    line([p3(1) p4(1)],[p3(2) p4(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])

    p1 = Tor*[+L*0.5 -l 1]';
    p2 = Tor*[-L*0.5 -l 1]';
    p3 = Tor*[+L*0.5 +l 1]';
    p4 = Tor*[-L*0.5 +l 1]';
    line([p1(1) p2(1)],[p1(2) p2(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])
    line([p1(1) p3(1)],[p1(2) p3(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])
    line([p2(1) p4(1)],[p2(2) p4(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])
    line([p3(1) p4(1)],[p3(2) p4(2)],'LineWidth',2,'MarkerSize',10,'color',[0 0 1])
  