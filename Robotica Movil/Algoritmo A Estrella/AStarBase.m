classdef AStarBase < handle
    properties (Access = private)
        map = []; % Map: 0 - free space, 1 - obstacles
        open = []; closed = []; start = []; goal = []; act = []; path = [];
    end

    methods
        function obj = AStarBase(map)
            obj.map = map;
        end

        function path = find(obj, start, goal) % start=[is; js], goal=[ig; jg]
            obj.start = start; obj.goal = goal; obj.path = [];
            obj.closed = []; % Empty closed list
            obj.open = struct('id', start, 'src', [0; 0], 'g', 0, ...
            'h', obj.heuristic(start)); % Initial open list

            if obj.map(start(1), start(2))~=0 || obj.map(goal(1), goal(2))~=0
                path = []; return; % Path not feasible!
            end

            while true % Search loop
                if isempty(obj.open), break; end % No path found :(

                obj.act = obj.open(1); % Get node from the ordered open list,
                obj.closed = [obj.closed, obj.act]; % add it to the closed list
                obj.open = obj.open(2:end); % and remove it from the open list.

                if obj.act.id(1)==obj.goal(1) && obj.act.id(2)==obj.goal(2)
                    % Path found :) Get the path from the closed list ...
                    p = obj.act.id; obj.path = [p]; ids = [obj.closed.id];
                    while sum(abs(p-start))~=0 % Follow src nodes to the start
                        p = obj.closed(ids(1,:)==p(1) & ids(2,:)==p(2)).src;
                        obj.path = [p, obj.path];
                    end
                    break;
                end

                neighbours = obj.getNodeNeighbours(obj.act.id);
                for i = 1:size(neighbours, 2)
                    n = neighbours(:,i);
                    % Add neighbour to the open list if it is not in the closed
                    % list and it is not an obstacle.
                    ids = [obj.closed.id]; z = ids(1,:)==n(1) & ids(2,:)==n(2);
                    if isempty(find(z, 1)) && ~obj.map(n(1), n(2))
                        obj.addNodeToOpenList(n);
                    end
                end
            end

            path = obj.path;
        end
        
        function addNodeToOpenList(obj, i)
            g = obj.act.g + obj.cost(i); % Path cost
            % Check if the node is in already the open list
            ids = [obj.open.id]; s = [];
            
            if ~isempty(ids)
                s = sum(abs(ids-repmat(i, 1, size(ids, 2))))==0;
            end
            
            if isempty(find(s, 1)) % Add new node to the open list
                node = struct('id', i, 'src', obj.act.id, ...
                'g', g, 'h', obj.heuristic(i));
                obj.open = [obj.open, node];
            else % Update the node in the open list if it has better score
                if g<obj.open(s).g
                    obj.open(s).g = g;
                    obj.open(s).src = obj.act.id;
                end
            end
            % Sort open list
            [~,i] = sortrows([[obj.open.g]+[obj.open.h]; obj.open.h].', [1,2]);
            obj.open = obj.open(i);
        end

        function n = getNodeNeighbours(obj, a)
%             n = [a(1)-1, a(1), a(1)+1, a(1); a(2), a(2)+1, a(2), a(2)-1];
            n = [a(1)-1, a(1)-1, a(1)-1, a(1), a(1)+1, a(1)+1, a(1)+1, a(1); ...
                 a(2)-1, a(2), a(2)+1, a(2)+1, a(2)+1, a(2), a(2)-1, a(2)-1];
            
            [h, w] = size(obj.map);
            n = n(:, n(1,:)>=1 & n(1,:)<=h & n(2,:)>=1 & n(2,:)<=w); % Bounds
        end

        function g = cost(obj, a)
%             g = sum(abs(a-obj.act.id)); % Manhattan distance
            g = sqrt(sum((a-obj.act.id).^2)); % Euclidean distance
        end

        function h = heuristic(obj, a)
%             h = sqrt(sum((a-obj.goal).^2)); % Euclidean distance
            h = sum(abs(a-obj.goal)); % Manhattan distance
        end
    end
end