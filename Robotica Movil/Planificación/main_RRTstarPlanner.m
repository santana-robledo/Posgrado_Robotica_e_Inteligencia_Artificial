clear all
close all
clc

%% Mapa
load occMap

figure
show(occMap)

inflate(occMap,0.2)

figure
show(occMap)

%% un click para inicio (x(1),y(1)), otro click para meta (x(2),y(2))
[x,y] = ginput(2);
x = round(x); y = round(y);

%% RRT*
ss = stateSpaceSE2;
ss.StateBounds = [occMap.XWorldLimits; occMap.YWorldLimits; [0 0]];

sv = validatorOccupancyMap(ss);
sv.Map = occMap;

planner = plannerRRTStar(ss,sv);
planner.ContinueAfterGoalReached = true;
planner.MaxIterations = 2500;
planner.MaxConnectionDistance = 0.5;

start = [x(1) y(1) 0];
goal = [x(2) y(2) 0];

[path,graph] = plan(planner,start,goal);


figure
show(occMap)
hold on
plot(graph.TreeData(:,1),graph.TreeData(:,2), '.-'); % tree expansion
plot(path.States(:,1),path.States(:,2),'r-','LineWidth',2); % draw path


path = path.States(:,1:2)'

save('RRT_path','path')
