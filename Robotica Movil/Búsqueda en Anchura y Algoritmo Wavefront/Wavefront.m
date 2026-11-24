classdef Wavefront < handle
    properties (Access = public)
        map = [];
        wf_map = [];
    end
    
    methods
        function obj = Wavefront(map)
            obj.map = map;
            obj.wf_map = map;
        end
        
        function path = Find (obj,start,goal)
            %% Update Map
            obj.wf_map = obj.map;
            [H,W] = size(obj.map);

            moves = [-1 0 +1 0; 0 +1 0 -1]; % Neumann
            [~,m] = size(moves);
            
            abiertos = goal;
            cerrados = [];
            vecinos = [];
            
            valor = 2;
            obj.wf_map(goal(1),goal(2)) = valor;
            
            if obj.map(goal(1),goal(2))==1 ||  obj.map(start(1),start(2))==1
                error('Rejilla ocupada')
            end

            while ~isempty(abiertos)
                while ~isempty(abiertos)
                    actual = abiertos(:,1); % Siempre retirar la primera rejilla
                    cerrados = [cerrados actual];
                    abiertos(:,1) = [];
            
                    obj.wf_map(actual(1),actual(2)) = valor; % Asignamos el valor a la rejilla actual
                    
                    % Revision vecindario
                    for k=1:m
                        vecino = actual + moves(:,k);
            
                        if ~(1<=vecino(1) && vecino(1)<=H) % dimensiones del mapa? (H)
                            continue
                        end
                            
                        if ~(1<=vecino(2) && vecino(2)<=W) % dimensiones del mapa? (W)
                            continue
                        end
    
                        if obj.map(vecino(1),vecino(2))==1 % rejilla ocupada?
                            continue
                        end

                        if ~(sum(1.0*(cerrados(1,:)==vecino(1)&cerrados(2,:)==vecino(2)))==0) % lista de cerrados?
                            continue
                        end
    
                        if ~(sum(1.0*(abiertos(1,:)==vecino(1)&abiertos(2,:)==vecino(2)))==0) % lista de abiertos?
                            continue
                        end

                        if isempty(vecinos)
                            vecinos = [vecinos vecino];
                        else
                            % lista de vecinos?
                            if sum(1.0*(vecinos(1,:)==vecino(1)&vecinos(2,:)==vecino(2)))==0
                                vecinos = [vecinos vecino];
                            end
                        end
                    end
                end
            
                abiertos = [abiertos vecinos];
                vecinos = [];
                
                valor = valor + 1;
            end

            %% Encontrar path
            moves = [-1 -1 -1 0 1 1 1 0; -1 0 1 1 1 0 -1 -1]; % Moore
            [~,m] = size(moves);
            
            actual = start;
            path = [];
            
            val = obj.wf_map(actual(1),actual(2));
            path = [path actual];
            
            % Actualizar el path hasta la rejilla con valor de 2
            while(val~=2)
                values = inf*ones(1,m);
                
                for k=1:m
                    vecino = actual + moves(:,k);
                    
                    if ~(1<=vecino(1) && vecino(1)<=H) % dimensiones del mapa? (H)
                        continue
                    end
                    
                    if ~(1<=vecino(2) && vecino(2)<=W) % dimensiones del mapa? (W)
                        continue
                    end
                    
                    if obj.map(vecino(1),vecino(2))==1 % rejilla ocupada?
                        continue
                    end

                    values(k) = obj.wf_map(vecino(1),vecino(2));
                end
                
                % k_best es el indice del vecino con el menor valor 
                [val,k_best] = min(values);
                vecino = actual + moves(:,k_best);
                
                path = [path vecino];
	            actual = vecino;
            end
        end
    end
end