classdef DR12 < handle
    properties (Access = private)
        CoppeliaSim = [];
        ClientID = [];
        
        motorL = [];
        motorR = [];
        S = [];

        d_max = 1.0;
        d_min = 0.105;
    end
    
    methods
        function obj = DR12()
            try
                obj.CoppeliaSim = remApi('remoteApi');
                obj.CoppeliaSim.simxFinish(-1);
                obj.ClientID = obj.CoppeliaSim.simxStart('127.0.0.1',19997,true,true,5000,5); 
                obj.CoppeliaSim.simxStopSimulation(obj.ClientID,obj.CoppeliaSim.simx_opmode_oneshot_wait);
                obj.CoppeliaSim.simxStartSimulation(obj.ClientID,obj.CoppeliaSim.simx_opmode_blocking);

                [~,obj.motorL] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/dr12/leftJoint_',obj.CoppeliaSim.simx_opmode_oneshot_wait);
                [~,obj.motorR] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/dr12/rightJoint_',obj.CoppeliaSim.simx_opmode_oneshot_wait);

                [~,obj.S(1)] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/dr12/S1',obj.CoppeliaSim.simx_opmode_oneshot_wait);
                [~,obj.S(2)] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/dr12/S2',obj.CoppeliaSim.simx_opmode_oneshot_wait);
                [~,obj.S(3)] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/dr12/S3',obj.CoppeliaSim.simx_opmode_oneshot_wait);
                [~,obj.S(4)] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/dr12/S4',obj.CoppeliaSim.simx_opmode_oneshot_wait);
                [~,obj.S(5)] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/dr12/S5',obj.CoppeliaSim.simx_opmode_oneshot_wait);
                [~,obj.S(6)] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/dr12/S6',obj.CoppeliaSim.simx_opmode_oneshot_wait);
                [~,obj.S(7)] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/dr12/S7',obj.CoppeliaSim.simx_opmode_oneshot_wait);
            catch E
                error(E)
            end
        end

        function obj = Set_Joint_Velocity (obj,wl,wr)
            try
                obj.CoppeliaSim.simxSetJointTargetVelocity(obj.ClientID,obj.motorL,wl,obj.CoppeliaSim.simx_opmode_streaming);
                obj.CoppeliaSim.simxSetJointTargetVelocity(obj.ClientID,obj.motorR,wr,obj.CoppeliaSim.simx_opmode_streaming);
            catch E
                error(E)
            end
        end

        function d = Read_Sensors(obj)
            try
                d = obj.d_max*ones(1,7);
                aux = ones(1,7);
    
                if obj.CoppeliaSim.simxGetConnectionId(obj.ClientID)==1
                    for j=1:7
                        [aux(j),aux_2,D] = obj.CoppeliaSim.simxReadProximitySensor(obj.ClientID,obj.S(j),obj.CoppeliaSim.simx_opmode_streaming);
    
                        if aux_2==1
                            d(j) = D(3);

                            if d(j)<=obj.d_min
                                d(j) = obj.d_min;
                            end
                        end
                    end
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