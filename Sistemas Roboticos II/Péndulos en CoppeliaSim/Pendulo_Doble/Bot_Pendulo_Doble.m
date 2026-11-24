classdef Bot_Pendulo_Doble < handle
    properties (Access = private)
        CoppeliaSim = [];
        ClientID = [];
        
        Joint_1 = [];
        Joint_2 = [];
    end
    
    methods
        function obj = Bot_Pendulo_Doble(dt)
            try
                obj.CoppeliaSim = remApi('remoteApi');
                obj.CoppeliaSim.simxFinish(-1);
                obj.ClientID = obj.CoppeliaSim.simxStart('127.0.0.1',19997,true,true,5000,5); 
                obj.CoppeliaSim.simxStopSimulation(obj.ClientID,obj.CoppeliaSim.simx_opmode_oneshot_wait);

                obj.CoppeliaSim.simxSetFloatingParameter(obj.ClientID,obj.CoppeliaSim.sim_floatparam_simulation_time_step, ...
                                                         dt,obj.CoppeliaSim.simx_opmode_oneshot);

                obj.CoppeliaSim.simxSynchronous(obj.ClientID,true);
                obj.CoppeliaSim.simxStartSimulation(obj.ClientID,obj.CoppeliaSim.simx_opmode_blocking);

                [~,obj.Joint_1] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/joint_1',obj.CoppeliaSim.simx_opmode_oneshot_wait);
                [~,obj.Joint_2] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/joint_2',obj.CoppeliaSim.simx_opmode_oneshot_wait);
            catch E
                rethrow(E)
            end
        end
        
        function q = Get_Position (obj)
            try
                aux = [1 1];
                while sum(aux)~=0
                    [aux(1),q1] = obj.CoppeliaSim.simxGetJointPosition(obj.ClientID,obj.Joint_1,obj.CoppeliaSim.simx_opmode_streaming);
                    [aux(2),q2] = obj.CoppeliaSim.simxGetJointPosition(obj.ClientID,obj.Joint_2,obj.CoppeliaSim.simx_opmode_streaming);
                end
            
                q = [double(q1); double(q2)];
            catch E
                rethrow(E)
            end
        end
        
        function qp = Get_Velocity (obj)
            try
                aux = [1 1];
                while sum(aux)~=0
                    [aux(1),qp_1] = obj.CoppeliaSim.simxGetObjectFloatParameter(obj.ClientID,obj.Joint_1,2012,obj.CoppeliaSim.simx_opmode_streaming);
                    [aux(2),qp_2] = obj.CoppeliaSim.simxGetObjectFloatParameter(obj.ClientID,obj.Joint_2,2012,obj.CoppeliaSim.simx_opmode_streaming);
                end
            
                qp = [double(qp_1); double(qp_2)];
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