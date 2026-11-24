classdef Bot_dr12 < handle
    properties (Access = private)
        CoppeliaSim = [];
        ClientID = [];
        camera = [];
        
        motorL = [];
        motorR = [];

        R = 0.086;
        L = 0.1635/2;
    end
    
    methods
        function obj = Bot_dr12()
            try
                obj.CoppeliaSim = remApi('remoteApi');
                obj.CoppeliaSim.simxFinish(-1);
                obj.ClientID = obj.CoppeliaSim.simxStart('127.0.0.1',19997,true,true,5000,5); 
                
                obj.CoppeliaSim.simxStopSimulation(obj.ClientID,obj.CoppeliaSim.simx_opmode_oneshot_wait);
                obj.CoppeliaSim.simxSynchronous(obj.ClientID,true);
                obj.CoppeliaSim.simxStartSimulation(obj.ClientID,obj.CoppeliaSim.simx_opmode_blocking);

                [~,obj.motorL] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/dr12/leftJoint_',obj.CoppeliaSim.simx_opmode_oneshot_wait);
                [~,obj.motorR] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/dr12/rightJoint_',obj.CoppeliaSim.simx_opmode_oneshot_wait);
                [~,obj.camera] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/kinect/rgb',obj.CoppeliaSim.simx_opmode_oneshot_wait);
            catch E
                rethrow(E)
            end
        end

        function obj = Set_Joint_Velocity (obj,v,w)
            try
                wr = (2*v+w*obj.L)/(2*obj.R);
                wl = (2*v-w*obj.L)/(2*obj.R);
                
                obj.CoppeliaSim.simxSetJointTargetVelocity(obj.ClientID,obj.motorL,wl,obj.CoppeliaSim.simx_opmode_streaming);
                obj.CoppeliaSim.simxSetJointTargetVelocity(obj.ClientID,obj.motorR,wr,obj.CoppeliaSim.simx_opmode_streaming);
            catch E
                rethrow(E)
            end
        end

        function img = Get_Image (obj)
            aux = 1;
            
            while aux~=0
                if obj.CoppeliaSim.simxGetConnectionId(obj.ClientID)==1
                    [aux,~,img] = obj.CoppeliaSim.simxGetVisionSensorImage2(obj.ClientID,obj.camera,0,obj.CoppeliaSim.simx_opmode_streaming);
                else
                    error(' connection lost...')
                end
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


            % try
            % 
            % catch E
            %     rethrow(E)
            % end