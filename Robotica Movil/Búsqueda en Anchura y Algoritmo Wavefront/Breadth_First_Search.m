classdef Breadth_First_Search < handle
    properties (Access = public)
        map = [];
    end
    
    methods
        function obj = Breadth_First_Search(map)
            obj.map = map;
        end

        function path = Find (obj,start,goal)
            % moves = [-1 0 +1 0; 0 +1 0 -1]; % Neumann
            moves = [-1 -1 -1 0 1 1 1 0; -1 0 1 1 1 0 -1 -1]; % Moore
            
            [~,m] = size(moves);
            
            a = []; % Lista de abiertos
            c = []; % Lista de cerrados
            
            s = []; % Lista de nodos padres
            t = []; % Lista de nodos hijos
            
            [H,W] = size(obj.map);
        
            a = [a start];
            
            if obj.map(goal(1),goal(2))==1 || obj.map(start(1),start(2))==1
                error('Rejilla ocupada')
            end

            while ~isempty(a)
                actual = a(:,1);
                
                c = [c actual];
                a(:,1) = [];
                
                for k=1:m
                    p = actual + moves(:,k);
                    
                    if ~(1<=p(1) && p(1)<=H) % dimensiones del mapa? (H)
                        continue
                    end
                        
                    if ~(1<=p(2) && p(2)<=W) % dimensiones del mapa? (W)
                        continue
                    end

                    if obj.map(p(1),p(2))==1 % rejilla ocupada?
                        continue
                    end

                    if ~(sum(1.0*(c(1,:)==p(1)&c(2,:)==p(2)))==0) % lista de cerrados?
                        continue
                    end

                    if ~(sum(1.0*(a(1,:)==p(1)&a(2,:)==p(2)))==0) % lista de abiertos?
                        continue
                    end

                    a = [a p];
                    
                    s = [s actual];
                    t = [t p];
                end
            end
            
            % Obtener path
            path = goal;
            s_node = goal;
            
            while sum(abs(s_node-start))~=0
                s_node = s(:,t(1,:)==s_node(1)&t(2,:)==s_node(2));
                path = [s_node path];
            end
        end
    end
end











