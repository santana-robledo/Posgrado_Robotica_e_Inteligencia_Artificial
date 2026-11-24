classdef Bot_Pioneer < handle
    properties (Access = private)
        CoppeliaSim = [];
        ClientID = [];
        
        Pioneer = [];
        motorL = [];
        motorR = [];

        usensors = [];
        d_max = 1.0;
        d_min = 0.1;

        Goal = [];
        Obs = [];
    end
    
    methods
        function obj = Bot_Pioneer()
            obj.CoppeliaSim = remApi('remoteApi');
            obj.CoppeliaSim.simxFinish(-1);
            disp('Connecting to robot....')
            obj.ClientID = obj.CoppeliaSim.simxStart('127.0.0.1',19997,true,true,5000,5); 
            obj.CoppeliaSim.simxStopSimulation(obj.ClientID,obj.CoppeliaSim.simx_opmode_oneshot_wait);

            obj.CoppeliaSim.simxSynchronous(obj.ClientID,true);
            obj.CoppeliaSim.simxStartSimulation(obj.ClientID,obj.CoppeliaSim.simx_opmode_blocking);

            if obj.ClientID==0
                [~,obj.Pioneer] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/PioneerP3DX',obj.CoppeliaSim.simx_opmode_oneshot_wait);
                [~,obj.motorL] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/PioneerP3DX/leftMotor',obj.CoppeliaSim.simx_opmode_oneshot_wait);
                [~,obj.motorR] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/PioneerP3DX/rightMotor',obj.CoppeliaSim.simx_opmode_oneshot_wait);
               
                for j=1:16
                    [~,obj.Obs(j)] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,['/80cmHighPillar25cm[' num2str(j-1) ']'],obj.CoppeliaSim.simx_opmode_oneshot_wait);
                end

                for i=1:16
                    [~,obj.usensors(i)] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,['/PioneerP3DX/ultrasonicSensor[' num2str(i-1) ']'],obj.CoppeliaSim.simx_opmode_oneshot_wait);
                end

                [~,obj.Goal] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/Goal',obj.CoppeliaSim.simx_opmode_oneshot_wait);

                disp(' done.')
            else
                error(' robot not connected...')
            end
        end
        
        function p = Get_Pose (obj)
            aux = ones(2,1);

            while sum(aux)~=0
                if obj.CoppeliaSim.simxGetConnectionId(obj.ClientID)==1
                    [aux(1),position] = obj.CoppeliaSim.simxGetObjectPosition(obj.ClientID,obj.Pioneer,-1,obj.CoppeliaSim.simx_opmode_streaming);
                    [aux(2),orientation] = obj.CoppeliaSim.simxGetObjectOrientation(obj.ClientID,obj.Pioneer,-1,obj.CoppeliaSim.simx_opmode_streaming);
                    
                    p = double([position(1) position(2) orientation(3)]');
                else
                    error(' connection lost...')
                end
            end
        end
        
        function obj = Set_Joint_Velocity (obj,w)
            if obj.CoppeliaSim.simxGetConnectionId(obj.ClientID)==1
                obj.CoppeliaSim.simxSetJointTargetVelocity(obj.ClientID,obj.motorR,w(1),obj.CoppeliaSim.simx_opmode_streaming);
                obj.CoppeliaSim.simxSetJointTargetVelocity(obj.ClientID,obj.motorL,w(2),obj.CoppeliaSim.simx_opmode_streaming);
            else
                error(' connection lost...')
            end
        end

        function p_goal = Get_Goal_Position (obj)
            paux = 1;

            while paux==1
                if obj.CoppeliaSim.simxGetConnectionId(obj.ClientID)==1
                    [paux,p] = obj.CoppeliaSim.simxGetObjectPosition(obj.ClientID,obj.Goal,-1,obj.CoppeliaSim.simx_opmode_streaming);
                    p = double(p');

                    p_goal = p(1:2);
                else
                    error(' connection lost...')
                end
            end
        end
        
        function p_obs = Get_Obs_Positions (obj)
            aux = ones(16,1);
            p_obs = zeros(2,16);

            while sum(aux)~=0
                if obj.CoppeliaSim.simxGetConnectionId(obj.ClientID)==1

                    for j=1:16
                        [aux(j),p] = obj.CoppeliaSim.simxGetObjectPosition(obj.ClientID,obj.Obs(j),-1,obj.CoppeliaSim.simx_opmode_streaming);
                        p = double(p');
                        p_obs(:,j) = p(1:2);
                    end       
               else
                    error(' connection lost...')
                end
            end
        end
        
        function [angles,ranges] = Get_Ultrasonic_Sensor_Scan (obj)
            aux = ones(1,16);
            ranges = zeros(1,16);
            angles = [90,50,30,10,-10,-30,-50,-90,-90,-130,-150,-170,170,150,130,90]*(pi/180);

            if obj.CoppeliaSim.simxGetConnectionId(obj.ClientID)==1
                for i=1:16
                    [~,aux(i),di] = obj.CoppeliaSim.simxReadProximitySensor(obj.ClientID,obj.usensors(i),obj.CoppeliaSim.simx_opmode_streaming);
                    ranges(i) = di(3);

                    if aux(i)>0 && ranges(i) <=obj.d_max
                        if ranges(i)<obj.d_min
                            ranges(i) = obj.d_min;
                        end
                    else
                        ranges(i) = obj.d_max;
                    end
                end
            else
                error(' connection lost...')
            end
        end

        function s = Connection(obj)
            s = obj.CoppeliaSim.simxGetConnectionId(obj.ClientID);
        end

        function Simulation_Step (obj)
            try
                obj.CoppeliaSim.simxSynchronousTrigger(obj.ClientID);  
            catch E
                error(E)
            end
        end

        function Stop_Simulation (obj)
            try
                obj.CoppeliaSim.simxStopSimulation(obj.ClientID,obj.CoppeliaSim.simx_opmode_oneshot_wait);
            catch E
                error(E)
            end
        end
    end
end