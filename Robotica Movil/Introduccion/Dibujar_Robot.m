function Dibujar_Robot (tipo, p, options)
    % p = [0.5; -0.5; pi/4];
    %
    % Dibujar_Robot('Omnidireccional_3w',p,L=0.3)
    %
    % Dibujar_Robot('Omnidireccional_4w',p,L=0.3)
    %
    % Dibujar_Robot('Omnidireccional_4wm',p,L1=0.25,l2=0.20)
    %
    % Dibujar_Robot('Coche',p,alpha=-pi/4,d=0.25)
    %
    % Dibujar_Robot('Triciclo',p,alpha=-pi/4,d=0.25)
    %
    % Dibujar_Robot('Monociclo_Desacoplado',p,d=0.25)
    %
    % Dibujar_Robot('Diferencial',p,L=0.20,R=0.10)
    %
    % Dibujar_Robot('Monociclo',p)
    %
    % Dibujar_Robot('Monociclo',p,Size=0.3)
    %

    arguments
        tipo (1,:) char {mustBeMember(tipo, {'Omnidireccional_3w', 'Omnidireccional_4w','Omnidireccional_4wm', 'Coche', 'Triciclo', 'Monociclo_Desacoplado', 'Diferencial', 'Monociclo'})}
        p (3,1) double
        
        options.Size (1,1) double = 0.2
        options.alpha (1,1) double
        options.d (1,1) double
        options.L (1,1) double
        options.L1 (1,1) double
        options.l2 (1,1) double
        options.R (1,1) double
    end

    switch tipo
        case 'Omnidireccional_3w'
            Dibujar_Omnidireccional_3w(p, options.L);
        case 'Omnidireccional_4w'
            Dibujar_Omnidireccional_4w(p, options.L);
        case 'Omnidireccional_4wm'
            Dibujar_Omnidireccional_4wm(p, options.L1, options.l2);
        case 'Coche'
            Dibujar_Coche(p, options.alpha, options.d);
        case 'Triciclo'
            Dibujar_Triciclo(p, options.alpha, options.d);
        case 'Monociclo_Desacoplado'
            Dibujar_Monociclo_Desacoplado(p, options.d);
        case 'Diferencial'
            Dibujar_Diferencial(p, options.L, options.R);
        case 'Monociclo'
            Dibujar_Monociclo(p, Size=options.Size);
        otherwise
            error('El modelo no existe')
    end

    zoom = 2;
    x_min = p(1) - zoom;
    x_max = p(1) + zoom;
    y_min = p(2) - zoom;
    y_max = p(2) + zoom;

    xlabel('x')
    ylabel('y')
    axis([x_min x_max y_min y_max]);
    axis equal
end

function Dibujar_Omnidireccional_3w (p,L)
    arguments
        p (3,1) double
        L (1,1) double
    end

    rueda_radio = 0.50*L;
    rueda_ancho = 0.20*L;
    Tb = [cos(p(3)) -sin(p(3)) p(1); sin(p(3)) cos(p(3)) p(2); 0 0 1];

    phi = linspace(0,2*pi,50);
    base_radio = L-rueda_ancho;
    base_cx = p(1) + base_radio*cos(phi);
    base_cy = p(2) + base_radio*sin(phi);
    
    T1 = [0 -1 base_radio+rueda_ancho; 1 0 0; 0 0 1];
    T2 = [cos(2*pi/3) -sin(2*pi/3) 0; sin(2*pi/3) cos(2*pi/3) 0; 0 0 1];

    hold on
    grid on
    Dibujar_Rueda(Tb*T1,rueda_radio,rueda_ancho);
    Dibujar_Rueda(Tb*T2*T1,rueda_radio,rueda_ancho);
    Dibujar_Rueda(Tb*T2'*T1,rueda_radio,rueda_ancho);
    plot(base_cx, base_cy, 'k', 'LineWidth', 2)

    phi = linspace(0,2*pi,50);
    p_frente = Tb*[0.6*base_radio; 0; 1];
    cx_frente = p_frente(1) + (0.15*base_radio)*cos(phi);
    cy_frente = p_frente(2) + (0.15*base_radio)*sin(phi);

    fill(cx_frente, cy_frente, 'r', 'EdgeColor', 'k', 'LineWidth', 2)
end

function Dibujar_Omnidireccional_4w (p,L)
    arguments
        p (3,1) double
        L (1,1) double
    end

    rueda_radio = 0.45*L;
    rueda_ancho = 0.20*L;
    Tb = [cos(p(3)) -sin(p(3)) p(1); sin(p(3)) cos(p(3)) p(2); 0 0 1];

    phi = linspace(0,2*pi,50);
    base_radio = L-rueda_ancho;
    base_cx = p(1) + base_radio*cos(phi);
    base_cy = p(2) + base_radio*sin(phi);
    
    T1 = [0 -1 base_radio+rueda_ancho; 1 0 0; 0 0 1];
    T2 = [cos(pi/4) -sin(pi/4) 0; sin(pi/4) cos(pi/4) 0; 0 0 1];
    T3 = [cos(3*pi/4) -sin(3*pi/4) 0; sin(3*pi/4) cos(3*pi/4) 0; 0 0 1];

    hold on
    grid on
    Dibujar_Rueda(Tb*T2*T1,rueda_radio,rueda_ancho);
    Dibujar_Rueda(Tb*T2'*T1,rueda_radio,rueda_ancho);
    Dibujar_Rueda(Tb*T3*T1,rueda_radio,rueda_ancho);
    Dibujar_Rueda(Tb*T3'*T1,rueda_radio,rueda_ancho);
    plot(base_cx, base_cy, 'k', 'LineWidth', 2)

    phi = linspace(0,2*pi,50);
    p_frente = Tb*[0.6*base_radio; 0; 1];
    cx_frente = p_frente(1) + (0.15*base_radio)*cos(phi);
    cy_frente = p_frente(2) + (0.15*base_radio)*sin(phi);

    fill(cx_frente, cy_frente, 'r', 'EdgeColor', 'k', 'LineWidth', 2)
end

function Dibujar_Omnidireccional_4wm (p,L,l)
    arguments
        p (3,1) double
        L (1,1) double
        l (1,1) double
    end

    rueda_radio = 0.3*L;
    rueda_ancho = 0.2*l;
    base_largo = 1.5*L;
    base_ancho = l-rueda_ancho;

    Tb = [cos(p(3)) -sin(p(3)) p(1); sin(p(3)) cos(p(3)) p(2); 0 0 1];
    T_fl = Tb*[1 0 L; 0 1 l; 0 0 1];
    T_fr = Tb*[1 0 L; 0 1 -l; 0 0 1];

    T_rl = Tb*[1 0 -L; 0 1 l; 0 0 1];
    T_rr = Tb*[1 0 -L; 0 1 -l; 0 0 1];

    phi = linspace(0,2*pi,50);
    p_frente = Tb*[L; 0; 1];
    cx_frente = p_frente(1) + (0.15*L)*cos(phi);
    cy_frente = p_frente(2) + (0.15*L)*sin(phi);

    hold on
    grid on
    Dibujar_Base(Tb,base_largo,base_ancho);
    Dibujar_Rueda(T_fl,rueda_radio,rueda_ancho);
    Dibujar_Rueda(T_fr,rueda_radio,rueda_ancho);
    Dibujar_Rueda(T_rl,rueda_radio,rueda_ancho);
    Dibujar_Rueda(T_rr,rueda_radio,rueda_ancho);
    fill(cx_frente, cy_frente, 'r', 'EdgeColor', 'k', 'LineWidth', 2)
end

function Dibujar_Coche (p,alpha,d)
    arguments
        p (3,1) double
        alpha (1,1) double
        d (1,1) double
    end

    base_largo = 1.1*d;
    base_ancho = 0.90*base_largo;
    rueda_radio = 0.25*base_largo;
    rueda_ancho = 0.20*base_ancho;
    desp_base = 0.45*base_largo;
    desp_ruedas = 0.40*base_largo;

    Tb = [cos(p(3)) -sin(p(3)) p(1); sin(p(3)) cos(p(3)) p(2); 0 0 1];
    Tl_rueda_delantera = [cos(alpha) -sin(alpha) d; sin(alpha) cos(alpha) desp_ruedas; 0 0 1];
    Tr_rueda_delantera = [cos(alpha) -sin(alpha) d; sin(alpha) cos(alpha) -desp_ruedas; 0 0 1];
    Tl = Tb*[1 0 0; 0 1 desp_ruedas; 0 0 1];
    Tr = Tb*[1 0 0; 0 1 -desp_ruedas; 0 0 1];

    hold on
    grid on
    Dibujar_Base(Tb*[1 0 desp_base; 0 1 0; 0 0 1],base_largo,base_ancho);
    Dibujar_Rueda(Tl,rueda_radio,rueda_ancho);
    Dibujar_Rueda(Tr,rueda_radio,rueda_ancho);
    Dibujar_Rueda_Orientada(Tb*Tl_rueda_delantera,rueda_radio,rueda_ancho);
    Dibujar_Rueda_Orientada(Tb*Tr_rueda_delantera,rueda_radio,rueda_ancho);
end

function Dibujar_Triciclo (p,alpha,d)
    arguments
        p (3,1) double
        alpha (1,1) double
        d (1,1) double
    end

    base_largo = 1.1*d;
    base_ancho = 0.50*base_largo;
    rueda_radio = 0.25*base_largo;
    rueda_ancho = 0.30*base_ancho;
    desp_base = 0.40*base_largo;

    Tb = [cos(p(3)) -sin(p(3)) p(1); sin(p(3)) cos(p(3)) p(2); 0 0 1];
    T_rueda_delantera = [cos(alpha) -sin(alpha) d; sin(alpha) cos(alpha) 0; 0 0 1];
    Tl = Tb*[1 0 0; 0 1 base_ancho+rueda_ancho; 0 0 1];
    Tr = Tb*[1 0 0; 0 1 -(base_ancho+rueda_ancho); 0 0 1];

    hold on
    grid on
    Dibujar_Base(Tb*[1 0 desp_base; 0 1 0; 0 0 1],base_largo,base_ancho);
    Dibujar_Rueda(Tl,rueda_radio,rueda_ancho);
    Dibujar_Rueda(Tr,rueda_radio,rueda_ancho);
    Dibujar_Rueda_Orientada(Tb*T_rueda_delantera,rueda_radio,rueda_ancho);
end

function Dibujar_Monociclo_Desacoplado (p,d)
    arguments
        p (3,1) double
        d (1,1) double
    end

    base_largo = 1.2*d;
    base_ancho = 0.7*base_largo;
    rueda_radio = 0.35*base_largo;
    rueda_ancho = 0.30*base_ancho;
    desp_base = 0.3*base_largo;

    Tb = [cos(p(3)) -sin(p(3)) p(1); sin(p(3)) cos(p(3)) p(2); 0 0 1];
    Tl = Tb*[1 0 0; 0 1 base_ancho+rueda_ancho; 0 0 1];
    Tr = Tb*[1 0 0; 0 1 -(base_ancho+rueda_ancho); 0 0 1];
    
    phi = linspace(0,2*pi,50);
    p_frente = Tb*[d; 0; 1];
    cx_frente = p_frente(1) + (0.15*base_largo)*cos(phi);
    cy_frente = p_frente(2) + (0.15*base_largo)*sin(phi);
    
    hold on
    grid on
    Dibujar_Base(Tb*[1 0 desp_base; 0 1 0; 0 0 1],base_largo,base_ancho);
    Dibujar_Rueda(Tl,rueda_radio,rueda_ancho);
    Dibujar_Rueda(Tr,rueda_radio,rueda_ancho);
    fill(cx_frente, cy_frente, 'r', 'EdgeColor', 'k', 'LineWidth', 2)
end

function Dibujar_Diferencial (p,L,R)
    arguments
        p (3,1) double
        L (1,1) double
        R (1,1) double
    end

    base_largo = 1.2*L;
    rueda_ancho = 0.5*R;
    rueda_radio = R;
    base_ancho = L - rueda_ancho;
    desp_base = 0.4*base_largo;

    Tb = [cos(p(3)) -sin(p(3)) p(1); sin(p(3)) cos(p(3)) p(2); 0 0 1];
    Tl = Tb*[1 0 0; 0 1 L; 0 0 1];
    Tr = Tb*[1 0 0; 0 1 -L; 0 0 1];
    
    phi = linspace(0,2*pi,50);
    p_frente = Tb*[1.0*base_largo; 0; 1];
    cx_frente = p_frente(1) + (0.15*base_largo)*cos(phi);
    cy_frente = p_frente(2) + (0.15*base_largo)*sin(phi);
    
    hold on
    grid on
    Dibujar_Base(Tb*[1 0 desp_base; 0 1 0; 0 0 1],base_largo,base_ancho);
    Dibujar_Rueda(Tl,rueda_radio,rueda_ancho);
    Dibujar_Rueda(Tr,rueda_radio,rueda_ancho);
    fill(cx_frente, cy_frente, 'r', 'EdgeColor', 'k', 'LineWidth', 2)
end

function Dibujar_Monociclo (p,options)
    arguments
        p (3,1) double
        options.Size (1,1) double = 0.2
    end

    base_largo = options.Size;
    base_ancho = 0.7*base_largo;
    rueda_radio = 0.40*base_largo;
    rueda_ancho = 0.35*base_ancho;
    desp_base = 0.3*base_largo;

    Tb = [cos(p(3)) -sin(p(3)) p(1); sin(p(3)) cos(p(3)) p(2); 0 0 1];
    Tl = Tb*[1 0 0; 0 1 base_ancho+rueda_ancho; 0 0 1];
    Tr = Tb*[1 0 0; 0 1 -(base_ancho+rueda_ancho); 0 0 1];
    
    phi = linspace(0,2*pi,50);
    p_frente = Tb*[0.85*base_largo; 0; 1];
    cx_frente = p_frente(1) + (0.15*base_largo)*cos(phi);
    cy_frente = p_frente(2) + (0.15*base_largo)*sin(phi);
    
    hold on
    grid on
    Dibujar_Base(Tb*[1 0 desp_base; 0 1 0; 0 0 1],base_largo,base_ancho);
    Dibujar_Rueda(Tl,rueda_radio,rueda_ancho);
    Dibujar_Rueda(Tr,rueda_radio,rueda_ancho);
    fill(cx_frente, cy_frente, 'r', 'EdgeColor', 'k', 'LineWidth', 2)
end

function Dibujar_Base (T,largo,ancho)
    P = T*[-largo largo largo -largo; ancho ancho -ancho -ancho; 1 1 1 1];
    plot([P(1,:), P(1,1)], [P(2,:), P(2,1)], 'k', 'LineWidth', 2);
end

function Dibujar_Rueda_Orientada (T,radio,ancho)
    R = T*[-radio radio radio -radio; ancho ancho -ancho -ancho; 1  1  1  1];
    fill(R(1,:), R(2,:), [0.3 0.3 0.3], 'EdgeColor', 'k', 'LineWidth', 2); 
    plot([R(1,2) R(1,3)], [R(2,2) R(2,3)], 'r', 'LineWidth', 2);
end

function Dibujar_Rueda (T,radio,ancho)
    P = T*[-radio radio radio -radio; ancho ancho -ancho -ancho; 1 1 1 1];
    fill(P(1,:), P(2,:), [0.5 0.5 0.5], 'EdgeColor', 'k', 'LineWidth', 2);
end
