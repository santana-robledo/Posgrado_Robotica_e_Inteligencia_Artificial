classdef Bot_Pioneer < handle
    properties (Access = private)
        CoppeliaSim = [];
        ClientID = [];
        
        Pioneer = [];
        motorL = [];
        motorR = [];

        R = 0.195/2;
        L = 0.381;

        is_synchronized = 1;
    end
    
    methods
        function obj = Bot_Pioneer(is_synchronized)
            try
                obj.CoppeliaSim = remApi('remoteApi');
                obj.CoppeliaSim.simxFinish(-1);
                obj.ClientID = obj.CoppeliaSim.simxStart('127.0.0.1',19997,true,true,5000,5); 
                
                obj.CoppeliaSim.simxStopSimulation(obj.ClientID,obj.CoppeliaSim.simx_opmode_oneshot_wait);
                
                if is_synchronized==1
                    obj.is_synchronized = is_synchronized;
                    obj.CoppeliaSim.simxSynchronous(obj.ClientID,true);
                end
                
                obj.CoppeliaSim.simxStartSimulation(obj.ClientID,obj.CoppeliaSim.simx_opmode_blocking);

                [~,obj.Pioneer] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/PioneerP3DX',obj.CoppeliaSim.simx_opmode_oneshot_wait);
                [~,obj.motorL] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/PioneerP3DX/leftMotor',obj.CoppeliaSim.simx_opmode_oneshot_wait);
                [~,obj.motorR] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/PioneerP3DX/rightMotor',obj.CoppeliaSim.simx_opmode_oneshot_wait);
                [~,~] = obj.CoppeliaSim.simxGetStringSignal(obj.ClientID,'measuredDataAtThisTime',obj.CoppeliaSim.simx_opmode_streaming);
            catch E
                rethrow(E)
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

        function [Angles,Ranges] = Get_LaserScans (obj)
            if obj.CoppeliaSim.simxGetConnectionId(obj.ClientID)==1
                [result,data] = obj.CoppeliaSim.simxGetStringSignal(obj.ClientID,'measuredDataAtThisTime',obj.CoppeliaSim.simx_opmode_streaming);

                if result~=0
                    Angles = [];
                    Ranges = [];
                else
                    laserData = obj.CoppeliaSim.simxUnpackFloats(data);
                    laserDataX = double(laserData(1:2:end-1));
                    laserDataY = double(laserData(2:2:end));
                    [Angles,Ranges] = cart2pol(laserDataX,laserDataY);
                end
            else
                error(' connection lost...')
            end
        end
        
        function obj = Set_Joint_Velocity (obj,v,w)
            if obj.CoppeliaSim.simxGetConnectionId(obj.ClientID)==1
                wr = (2*v+w*obj.L)/(2*obj.R);
                wl = (2*v-w*obj.L)/(2*obj.R);

                obj.CoppeliaSim.simxSetJointTargetVelocity(obj.ClientID,obj.motorR,wr,obj.CoppeliaSim.simx_opmode_streaming);
                obj.CoppeliaSim.simxSetJointTargetVelocity(obj.ClientID,obj.motorL,wl,obj.CoppeliaSim.simx_opmode_streaming);
            else
                error(' connection lost...')
            end
        end

        function Simulation_Step (obj)
            try
                if obj.is_synchronized==1
                    obj.CoppeliaSim.simxSynchronousTrigger(obj.ClientID);
                end
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