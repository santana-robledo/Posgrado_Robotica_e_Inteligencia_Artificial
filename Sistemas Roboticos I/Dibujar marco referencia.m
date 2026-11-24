function Dibujar_Marco_Referencia (T,options)
    arguments
        T double
        options.Alpha {mustBeNumeric} = 0.4
        options.LineStyle string = '-'
    end


    px = T*[1.5*options.Alpha 0 0 1]';
    py = T*[0 1.5*options.Alpha 0 1]';
    pz = T*[0 0 1.5*options.Alpha 1]';
    
    t = T(1:3,4);
    
    line([t(1) px(1)],[t(2) px(2)],[t(3) px(3)],'color',[1 0 0],'LineWidth',2,'LineStyle',options.LineStyle)
    line([t(1) py(1)],[t(2) py(2)],[t(3) py(3)],'color',[0 1 0],'LineWidth',2,'LineStyle',options.LineStyle)
    line([t(1) pz(1)],[t(2) pz(2)],[t(3) pz(3)],'color',[0 0 1],'LineWidth',2,'LineStyle',options.LineStyle)

    text(px(1),px(2),px(3),'x','Color','blue','FontSize',15)
    text(py(1),py(2),py(3),'y','Color','blue','FontSize',15)
    text(pz(1),pz(2),pz(3),'z','Color','blue','FontSize',15)

    hold on
    grid on
    axis([-2 2 -2 2 -2 2])
    view([-40 40])
    xlabel('x')
    ylabel('y')
    zlabel('z')
end