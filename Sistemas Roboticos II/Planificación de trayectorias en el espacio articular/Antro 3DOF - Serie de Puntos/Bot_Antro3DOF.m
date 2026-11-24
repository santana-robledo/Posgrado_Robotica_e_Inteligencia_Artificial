classdef Bot_Antro3DOF < handle
    properties (Access = private)
        CoppeliaSim = [];
        ClientID = [];
        
        Joint_1 = [];
        Joint_2 = [];
        Joint_3 = [];

        References = [-1 -1 -1 -1];
    end
    
    methods
        function obj = Bot_Antro3DOF(dt)
            try
                obj.CoppeliaSim = remApi('remoteApi');
                obj.CoppeliaSim.simxFinish(-1);
                obj.ClientID = obj.CoppeliaSim.simxStart('127.0.0.1',19997,true,true,5000,5); 
                obj.CoppeliaSim.simxStopSimulation(obj.ClientID,obj.CoppeliaSim.simx_opmode_oneshot_wait);

                obj.CoppeliaSim.simxSetFloatingParameter(obj.ClientID,obj.CoppeliaSim.sim_floatparam_simulation_time_step, ...
                                                         dt,obj.CoppeliaSim.simx_opmode_oneshot);

                obj.CoppeliaSim.simxSynchronous(obj.ClientID,true);
                obj.CoppeliaSim.simxStartSimulation(obj.ClientID,obj.CoppeliaSim.simx_opmode_blocking);

                [~,obj.Joint_1] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/q1',obj.CoppeliaSim.simx_opmode_oneshot_wait);
                [~,obj.Joint_2] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/q2',obj.CoppeliaSim.simx_opmode_oneshot_wait);
                [~,obj.Joint_3] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/q3',obj.CoppeliaSim.simx_opmode_oneshot_wait);
            
                for i=1:4
                    [~,obj.References(i)] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,['/Reference[' num2str(i-1) ']'],obj.CoppeliaSim.simx_opmode_oneshot_wait);
                end
            catch E
                rethrow(E)
            end
        end
        
        function Pd = Get_References (obj)
            Pd = zeros(3,4);

            try
                for i=1:4
                    paux = 1;
                    
                    while paux==1
                        [paux,p] = obj.CoppeliaSim.simxGetObjectPosition(obj.ClientID,obj.References(i),-1,obj.CoppeliaSim.simx_opmode_streaming);
                        pd = double(p)'; 
                    end

                    Pd(:,i) = pd;
                end

            catch E
                rethrow(E)
            end
        end


        function q = Get_Position (obj)
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
        
        function qp = Get_Velocity (obj)
            try
                aux = [1 1 1];
                while sum(aux)~=0
                    [aux(1),qp_1] = obj.CoppeliaSim.simxGetObjectFloatParameter(obj.ClientID,obj.Joint_1,2012,obj.CoppeliaSim.simx_opmode_streaming);
                    [aux(2),qp_2] = obj.CoppeliaSim.simxGetObjectFloatParameter(obj.ClientID,obj.Joint_2,2012,obj.CoppeliaSim.simx_opmode_streaming);
                    [aux(3),qp_3] = obj.CoppeliaSim.simxGetObjectFloatParameter(obj.ClientID,obj.Joint_3,2012,obj.CoppeliaSim.simx_opmode_streaming);
                end
            
                qp = [double(qp_1); double(qp_2); double(qp_3)];
            catch E
                rethrow(E)
            end
        end

        function obj = Set_Force (obj,tau)
            try
                obj.CoppeliaSim.simxSetJointTargetVelocity(obj.ClientID,obj.Joint_1,sign(tau(1))*1e6,obj.CoppeliaSim.simx_opmode_streaming);
                obj.CoppeliaSim.simxSetJointForce(obj.ClientID,obj.Joint_1,abs(tau(1)),obj.CoppeliaSim.simx_opmode_streaming);

                obj.CoppeliaSim.simxSetJointTargetVelocity(obj.ClientID,obj.Joint_2,sign(tau(2))*1e6,obj.CoppeliaSim.simx_opmode_streaming);
                obj.CoppeliaSim.simxSetJointForce(obj.ClientID,obj.Joint_2,abs(tau(2)),obj.CoppeliaSim.simx_opmode_streaming);

                obj.CoppeliaSim.simxSetJointTargetVelocity(obj.ClientID,obj.Joint_3,sign(tau(3))*1e6,obj.CoppeliaSim.simx_opmode_streaming);
                obj.CoppeliaSim.simxSetJointForce(obj.ClientID,obj.Joint_3,abs(tau(3)),obj.CoppeliaSim.simx_opmode_streaming);
            catch E
                rethrow(E)
            end
        end

        function Simulation_Step (obj)
            try
                obj.CoppeliaSim.simxSynchronousTrigger(obj.ClientID);  
            catch E
                rethrow(E)
            end
        end

        function Stop_Simulation (obj)
            try
                obj.CoppeliaSim.simxStopSimulation(obj.ClientID,obj.CoppeliaSim.simx_opmode_oneshot_wait);
            catch E
                rethrow(E)
            end
        end
    end
end