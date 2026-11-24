classdef Bot_youBot_Platform < handle
    properties (Access = private)
        CoppeliaSim = [];
        ClientID = [];
        
        youBot = [];
        motor1 = [];
        motor2 = [];
        motor3 = [];
        motor4 = [];
    end
    
    methods
        function obj = Bot_youBot_Platform()
            try 
                obj.CoppeliaSim = remApi('remoteApi');
                obj.CoppeliaSim.simxFinish(-1);
                obj.ClientID = obj.CoppeliaSim.simxStart('127.0.0.1',19997,true,true,5000,5); 
                
                obj.CoppeliaSim.simxStopSimulation(obj.ClientID,obj.CoppeliaSim.simx_opmode_oneshot_wait);
                obj.CoppeliaSim.simxSynchronous(obj.ClientID,true);
                obj.CoppeliaSim.simxStartSimulation(obj.ClientID,obj.CoppeliaSim.simx_opmode_blocking);

                [~,obj.youBot] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/youBot',obj.CoppeliaSim.simx_opmode_oneshot_wait);
                [~,obj.motor1] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/youBot/rollingJoint_fl',obj.CoppeliaSim.simx_opmode_oneshot_wait);
                [~,obj.motor2] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/youBot/rollingJoint_fr',obj.CoppeliaSim.simx_opmode_oneshot_wait);
                [~,obj.motor3] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/youBot/rollingJoint_rl',obj.CoppeliaSim.simx_opmode_oneshot_wait);
                [~,obj.motor4] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/youBot/rollingJoint_rr',obj.CoppeliaSim.simx_opmode_oneshot_wait);
            
            
            
            catch E
                error(E)
            end
        end
        
        function p = Get_Pose (obj)
            try
                aux = ones(2,1);
    
                while sum(aux)~=0
                    if obj.CoppeliaSim.simxGetConnectionId(obj.ClientID)==1
                        [aux(1),position] = obj.CoppeliaSim.simxGetObjectPosition(obj.ClientID,obj.youBot,-1,obj.CoppeliaSim.simx_opmode_streaming);
                        [aux(2),orientation] = obj.CoppeliaSim.simxGetObjectOrientation(obj.ClientID,obj.youBot,-1,obj.CoppeliaSim.simx_opmode_streaming);
                        
                        p = double([position(1) position(2) orientation(3)]');
                    else
                        error(' connection lost...')
                    end
                end
            catch E
                error(E)
            end
        end
        
        function obj = Set_Joint_Velocity (obj,v)
            try
                R = 0.05;
                w = v/R;

                obj.CoppeliaSim.simxSetJointTargetVelocity(obj.ClientID,obj.motor1,-w(1),obj.CoppeliaSim.simx_opmode_streaming);
                obj.CoppeliaSim.simxSetJointTargetVelocity(obj.ClientID,obj.motor2,-w(2),obj.CoppeliaSim.simx_opmode_streaming);
                obj.CoppeliaSim.simxSetJointTargetVelocity(obj.ClientID,obj.motor3,-w(3),obj.CoppeliaSim.simx_opmode_streaming);
                obj.CoppeliaSim.simxSetJointTargetVelocity(obj.ClientID,obj.motor4,-w(4),obj.CoppeliaSim.simx_opmode_streaming);
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