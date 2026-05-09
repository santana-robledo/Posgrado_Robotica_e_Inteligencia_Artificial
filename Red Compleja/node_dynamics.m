function f = node_dynamics(x, type)

switch type
    case 1
        % Lorenz
        sigma = 10;
        rho = 28;
        beta = 8/3;
        
        f = [ sigma*(x(2)-x(1));
              x(1)*(rho-x(3)) - x(2);
              x(1)*x(2) - beta*x(3) ];
          
    case 2
        % Rossler
        a = 0.2;
        b = 0.2;
        c = 5.7;
        
        f = [ -x(2)-x(3);
               x(1)+a*x(2);
               b + x(3)*(x(1)-c) ];
           
    otherwise
        error('Tipo de nodo no reconocido');
end
end