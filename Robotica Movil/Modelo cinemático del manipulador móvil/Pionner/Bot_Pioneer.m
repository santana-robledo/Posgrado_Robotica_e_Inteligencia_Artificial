classdef Bot_Pioneer < handle
    properties (Access = private)
        CoppeliaSim = [];
        ClientID = [];
        
        Pioneer = [];
        motorL = [];
        motorR = [];

        Joint_1 = [];
        Joint_2 = [];
        Joint_3 = [];

        Reference = [];
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

                [~,obj.Joint_1] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/PioneerP3DX/q1',obj.CoppeliaSim.simx_opmode_oneshot_wait);
                [~,obj.Joint_2] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/PioneerP3DX/q2',obj.CoppeliaSim.simx_opmode_oneshot_wait);
                [~,obj.Joint_3] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/PioneerP3DX/q3',obj.CoppeliaSim.simx_opmode_oneshot_wait);

                [~,obj.Reference] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/ReferenceFrame',obj.CoppeliaSim.simx_opmode_oneshot_wait);
            catch E
                error(E)
            end
        end
        
        function pd = Get_Reference (obj)
            try
                paux = 1;

                while paux==1
                    [paux,p] = obj.CoppeliaSim.simxGetObjectPosition(obj.ClientID,obj.Reference,-1,obj.CoppeliaSim.simx_opmode_streaming);
                    pd = double(p)'; 
                end
            catch E
                rethrow(E)
            end
        end

        function q = Get_Arm_Position (obj)
            try
                aux = [1 1 1];
                while sum(aux)~=0
                    [aux(1),q1] = obj.CoppeliaSim.simxGetJointPosition(obj.ClientID,obj.Joint_1,obj.CoppeliaSim.simx_opmode_streaming);
                    [aux(2),q2] = obj.CoppeliaSim.simxGetJointPosition(obj.ClientID,obj.Joint_2,obj.CoppeliaSim.simx_opmode_streaming);
                    [aux(3),q3] = obj.CoppeliaSim.simxGetJointPosition(obj.ClientID,obj.Joint_3,obj.CoppeliaSim.simx_opmode_streaming);
                end

                q = [double(q1); double(q2); double(q3)];
            catch E
                rethrow(E)
            end
        end

        function obj = Set_Joint_Velocity (obj,qp)
            try
                obj.CoppeliaSim.simxSetJointTargetVelocity(obj.ClientID,obj.Joint_1,qp(1),obj.CoppeliaSim.simx_opmode_streaming);
                obj.CoppeliaSim.simxSetJointTargetVelocity(obj.ClientID,obj.Joint_2,qp(2),obj.CoppeliaSim.simx_opmode_streaming);
                obj.CoppeliaSim.simxSetJointTargetVelocity(obj.ClientID,obj.Joint_3,qp(3),obj.CoppeliaSim.simx_opmode_streaming);
            catch E
                rethrow(E)
            end
        end

        function p = Get_Platform_Pose (obj)
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
        
        function obj = Set_Platform_Velocity (obj,w)
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