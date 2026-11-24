classdef TurtleBot3_WafflePI < handle
    properties (Access = private)
        bot_n = [];

        % Camera
        camSub = [];

        % Lidar
        lidarSub = [];

        % Odometry
        odomSub = [];
        resetPub = [];
        resetMsg = [];

        % Velocity Command
        velPub = [];
        velMsg = [];
    end
    
    methods
        function obj = TurtleBot3_WafflePI(bot_ip)
            try
                disp('Connecting to TurtleBot3 Waffle PI...')
                n = str2double(bot_ip(20:21));
                obj.bot_n = num2str(n);

                rosshutdown
                rosinit(bot_ip)
                
                % Camera
                obj.camSub = rossubscriber(['/bot' obj.bot_n '/raspicam_node/image/compressed'],'sensor_msgs/CompressedImage');
                
                % Lidar
                obj.lidarSub = rossubscriber(['/bot' obj.bot_n '/scan'],'sensor_msgs/LaserScan');
                
                % Odometry
                obj.odomSub = rossubscriber(['/bot' obj.bot_n '/odom'],'nav_msgs/Odometry');
                obj.resetPub = rospublisher(['/bot' obj.bot_n '/reset'],'std_msgs/Empty');
                obj.resetMsg = rosmessage(obj.resetPub);

                % Velocity Command
                obj.velPub = rospublisher(['/bot' obj.bot_n '/cmd_vel'],"geometry_msgs/Twist") ;
                obj.velMsg = rosmessage(obj.velPub);

                disp(' done.')
            catch E
                error(E)
            end
        end

        %% Camera
        function img = Get_Image (obj)
            try 
                camMsg = receive(obj.camSub);
                camMsg.Format = 'bgr8; jpeg compressed bgr8';
            
                img = readImage(camMsg);
            catch E
                error(E)
            end
        end

        %% Lidar
        function [Angles,Ranges] = Get_LaserScans (obj)
            try
                scanMsg = receive(obj.lidarSub);

                Angles = (scanMsg.AngleMin:scanMsg.AngleIncrement:scanMsg.AngleMax)';
                Ranges = scanMsg.Ranges;

                I = Ranges==0;
                Angles(I) = [];
                Ranges(I) = [];
            catch E
                error(E)
            end
        end

        %% Odometry
        function Reset_Odometry (obj)
            try
                disp('Restarting odometry...')
                send(obj.resetPub,obj.resetMsg)
                pause(0.1)

                disp(' done.')
            catch E
                error(E)
            end
        end

        function p = Get_Pose (obj)
            try
                odomMsg = receive(obj.odomSub);
                pose = odomMsg.Pose.Pose;
                x = pose.Position.X;
                y = pose.Position.Y;
                
                quat = pose.Orientation;
                angles = quat2eul([quat.W quat.X quat.Y quat.Z]);
                theta = angles(1);  
                
                p = [x y theta]';
            catch E
                error(E)
            end
        end

        %% Velocity Command
        function Set_Velocity (obj,v,w)
            try
                v = max(v,-0.2); v = min(v,0.2);
                w = max(w,-0.9); w = min(w,0.9);

                obj.velMsg.Linear.X = v;
                obj.velMsg.Angular.Z = w;

                send(obj.velPub,obj.velMsg)
            catch E
                error(E)
            end
        end
    end
end