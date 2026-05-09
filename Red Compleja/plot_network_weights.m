function plot_network_weights(A, W, topology, nodeTypes)

% Asegurar simetria para red no dirigida
A = double(A ~= 0);
A = max(A, A');
W = (W + W')/2;
W = W .* A;

figure('Color','w');
G = graph(W);

if strcmpi(topology,'ring') || strcmpi(topology,'regular')
    p = plot(G, 'Layout', 'circle', 'NodeLabel', 1:size(A,1));
else
    p = plot(G, 'Layout', 'force', 'NodeLabel', 1:size(A,1));
end

title(['Red compleja con pesos - Topologia: ', topology]);

weights = G.Edges.Weight;

if isempty(weights)
    return;
end

% Escalado del grosor de aristas
if max(weights) == min(weights)
    LWidths = 2*ones(size(weights));
else
    LWidths = 0.5 + 4*(weights - min(weights)) / (max(weights)-min(weights));
end

p.LineWidth = LWidths;
p.MarkerSize = 8;

% Etiquetas de aristas con peso
labeledge(p, 1:numedges(G), round(weights,2));

% Colorear nodos segun tipo
for i = 1:length(nodeTypes)
    if nodeTypes(i) == 1
        highlight(p, i, 'NodeColor', [0 0.4470 0.7410]); % Lorenz
    elseif nodeTypes(i) == 2
        highlight(p, i, 'NodeColor', [0.8500 0.3250 0.0980]); % Rossler
    else
        highlight(p, i, 'NodeColor', [0.5 0.5 0.5]);
    end
end
end