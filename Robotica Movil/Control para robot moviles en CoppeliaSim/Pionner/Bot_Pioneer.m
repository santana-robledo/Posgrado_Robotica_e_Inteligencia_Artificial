classdef Bot_Pioneer < handle
    properties (Access = private)
        CoppeliaSim = [];
        ClientID = [];
        
        Pioneer = [];
        motorL = [];
        motorR = [];
    end
    
    methods
        function obj = Bot_Pioneer()
            try
                obj.CoppeliaSim = remApi('remoteApi');
                obj.CoppeliaSim.simxFinish(-1);
                obj.ClientID = obj.CoppeliaSim.simxStart('127.0.0.1',19997,true,true,5000,5); 
                obj.CoppeliaSim.simxStopSimulation(obj.ClientID,obj.CoppeliaSim.simx_opmode_oneshot_wait);

                obj.CoppeliaSim.simxSynchronous(obj.ClientID,true);
                obj.CoppeliaSim.simxStartSimulation(obj.ClientID,obj.CoppeliaSim.simx_opmode_blocking);

                [~,obj.Pioneer] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/PioneerP3DX',obj.CoppeliaSim.simx_opmode_oneshot_wait);
                [~,obj.motorL] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/PioneerP3DX/leftMotor',obj.CoppeliaSim.simx_opmode_oneshot_wait);
                [~,obj.motorR] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/PioneerP3DX/rightMotor',obj.CoppeliaSim.simx_opmode_oneshot_wait);

                
            catch E
                error(E)
            end
        end
        
        function p = Get_Pose (obj)
            try
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
            catch E
                error(E)
            end
        end
        
        function obj = Set_Joint_Velocity (obj,w)
            try
                obj.CoppeliaSim.simxSetJointTargetVelocity(obj.ClientID,obj.motorR,w(1),obj.CoppeliaSim.simx_opmode_streaming);
                obj.CoppeliaSim.simxSetJointTargetVelocity(obj.ClientID,obj.motorL,w(2),obj.CoppeliaSim.simx_opmode_streaming);
            catch E
                error(E)
            end
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