classdef Bot_KUKA_iiwa < handle
    properties (Access = private)
        CoppeliaSim = [];
        ClientID = [];
        
        Joints = -1*zeros(7,1);
    end
    
    methods
        function obj = Bot_KUKA_iiwa(dt)
            try
                obj.CoppeliaSim = remApi('remoteApi');
                obj.CoppeliaSim.simxFinish(-1);
                obj.ClientID = obj.CoppeliaSim.simxStart('127.0.0.1',19997,true,true,5000,5); 
                obj.CoppeliaSim.simxStopSimulation(obj.ClientID,obj.CoppeliaSim.simx_opmode_oneshot_wait);

                obj.CoppeliaSim.simxSetFloatingParameter(obj.ClientID,obj.CoppeliaSim.sim_floatparam_simulation_time_step, ...
                                                         dt,obj.CoppeliaSim.simx_opmode_oneshot);

                obj.CoppeliaSim.simxSynchronous(obj.ClientID,true);
                obj.CoppeliaSim.simxStartSimulation(obj.ClientID,obj.CoppeliaSim.simx_opmode_blocking);

                for i=1:7
                    [~,obj.Joints(i)] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,['/LBRiiwa7R800/q' num2str(i)],obj.CoppeliaSim.simx_opmode_oneshot_wait);
                end
            catch E
                rethrow(E)
            end
        end

        function q = Get_Joint_Position (obj)
            try
                q = zeros(7,1);
                aux = ones(1,7);

                while sum(aux)~=0
                    for i=1:7
                        [aux(i),q(i)] = obj.CoppeliaSim.simxGetJointPosition(obj.ClientID,obj.Joints(i),obj.CoppeliaSim.simx_opmode_streaming);
                    end
                end
            
                q = double(q);
            catch E
                rethrow(E)
            end
        end
        
        function qp = Get_Joint_Velocity (obj)
            try
                qp = zeros(7,1);
                aux = ones(1,7);

                while sum(aux)~=0
                    for i=1:7
                        [aux(i),qp(i)] = obj.CoppeliaSim.simxGetObjectFloatParameter(obj.ClientID,obj.Joints(i),2012,obj.CoppeliaSim.simx_opmode_streaming);
                    end
                end
            
                qp = double(qp);
            catch E
                rethrow(E)
            end
        end

        function obj = Set_Force (obj,tau)
            try
                for i=1:7
                    obj.CoppeliaSim.simxSetJointTargetVelocity(obj.ClientID,obj.Joints(i),sign(tau(i))*1e6,obj.CoppeliaSim.simx_opmode_streaming);
                    obj.CoppeliaSim.simxSetJointForce(obj.ClientID,obj.Joints(i),abs(tau(i)),obj.CoppeliaSim.simx_opmode_streaming);             
                end
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