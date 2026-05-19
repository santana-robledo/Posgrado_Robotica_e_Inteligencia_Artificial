function plot_network_weights(A, W, topology, nodeTypes)
    % Crear el grafo
    G = graph(W); 
    
    figure('Color','w', 'Name', 'Figura 1: Pesos de la Red');
    
    % Layout según la topología
    if strcmpi(topology,'ring') || strcmpi(topology,'regular') || strcmpi(topology,'exponential')
        p = plot(G, 'Layout', 'circle', 'NodeLabel', 1:size(A,1));
    else
        p = plot(G, 'Layout', 'force', 'NodeLabel', 1:size(A,1));
    end
    
    title(['Topología: ', topology, ' (Etiquetas = Pesos)']);
    
    % Obtener los pesos
    weights = G.Edges.Weight;
    
    if ~isempty(weights)
        % 1. Mostrar pesos numéricos sobre las aristas
        p.EdgeLabel = string(round(weights, 2));
        
        % 2. Color de línea sólido (Gris oscuro para que se lean bien los números)
        p.EdgeColor = [0.2 0.2 0.2]; 
        
        % 3. Grosor de línea proporcional al peso (Visualización estructural)
        % Si prefieres que todas las líneas sean iguales, comenta la siguiente línea:
        p.LineWidth = 0.5 + 4 * (weights - min(weights)) / (max(weights) - min(weights) + eps);
    end
    
    % Estética de nodos y etiquetas
    p.MarkerSize = 9;
    p.NodeFontSize = 11;
    p.EdgeFontSize = 9; % Tamaño de la fuente de los pesos
    
    % Diferenciar tipos de nodos por color de marcador
    for i = 1:length(nodeTypes)
        if nodeTypes(i) == 1
            highlight(p, i, 'NodeColor', [0 0.4470 0.7410]); % Lorenz (Azul)
        else
            highlight(p, i, 'NodeColor', [0.8500 0.3250 0.0980]); % Rossler (Naranja)
        end
    end
    
    axis tight off;
end