clear all
close all
clc

%%
global v w; % para control remoto....
v = 0; w = 0;

h_fig = figure;
set(h_fig,'KeyPressFcn',@myfun);

bot = Bot_Pioneer(0); % 1: Sincronizar, 0: No sincronizar
bot.Set_Joint_Velocity(v,w); % Set_Joint_Velocity(v,w)

% SLAM
maxLidarRange = 5; % en metros
mapResolution = 20; % Para mapa en rejillas de ocupacion (pixeles/metro) 
slamAlg = lidarSLAM(mapResolution,maxLidarRange);

% The following loop closure parameters are set empirically: 
% Higher loop closure threshold helps reject false positives
slamAlg.LoopClosureThreshold = 400; % default: 100, [50 500]
% Tune this distance based on your environment and the expected vehicle trajectory.
slamAlg.LoopClosureSearchRadius = 3; % default: 8, [3 15]
% Minimum change in pose required to process scans 
slamAlg.MovementThreshold = [0.1 0.1]; % default: [0 0], [translation rotation]

S = 60;
dt = 0.05;


for t=dt:dt:S
    pa = bot.Get_Pose();
    disp('p from get_pose()')
    pa'
    

    % SLAM inicial
    [Angles,Ranges] = bot.Get_LaserScans();

	scans = lidarScan(double(Ranges),double(Angles));
	addScan(slamAlg,scans);

    [~,poses] = scansAndPoses(slamAlg);
    pb = poses(end,:)';

    disp('p from SLAM()')
    pb'

    % Slam acaba


    bot.Set_Joint_Velocity(v,w);
    bot.Simulation_Step();


    cla
    hold on
    grid on
    show(slamAlg);
    title(['Simulation time = ' num2str(t)])
    drawnow
end

close
bot.Stop_Simulation();

%% Resultados
figure
show(slamAlg);
title({'Map of the Environment','Pose Graph'})

[scansSLAM,poses] = scansAndPoses(slamAlg);
occMap = buildMap(scansSLAM,poses,mapResolution,maxLidarRange);

figure
show(occMap);
title('Occupancy Map Environment')

save('occMap.mat','occMap')


%% Funciones
function myfun(~,event)
	global v w;
    
	switch event.Key
        case 'leftarrow'
            w = 0.3;
        case 'rightarrow'
            w = -0.3;
        case 'uparrow'
            v = 0.2;
        case 'downarrow'
            v = -0.2;
        otherwise
            v = 0; w = 0;
	end    
end
