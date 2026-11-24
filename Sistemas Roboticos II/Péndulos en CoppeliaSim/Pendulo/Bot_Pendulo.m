classdef Bot_Pendulo < handle
    properties (Access = private)
        CoppeliaSim = [];
        ClientID = [];
        
        Joint = [];
    end
    
    methods
        function obj = Bot_Pendulo(dt)
            try
                obj.CoppeliaSim = remApi('remoteApi');
                obj.CoppeliaSim.simxFinish(-1);
                obj.ClientID = obj.CoppeliaSim.simxStart('127.0.0.1',19997,true,true,5000,5); 
                obj.CoppeliaSim.simxStopSimulation(obj.ClientID,obj.CoppeliaSim.simx_opmode_oneshot_wait);

                obj.CoppeliaSim.simxSetFloatingParameter(obj.ClientID,obj.CoppeliaSim.sim_floatparam_simulation_time_step, ...
                                                         dt,obj.CoppeliaSim.simx_opmode_oneshot);
                
                obj.CoppeliaSim.simxSynchronous(obj.ClientID,true);
                obj.CoppeliaSim.simxStartSimulation(obj.ClientID,obj.CoppeliaSim.simx_opmode_blocking);

                [~,obj.Joint] = obj.CoppeliaSim.simxGetObjectHandle(obj.ClientID,'/joint',obj.CoppeliaSim.simx_opmode_oneshot_wait);
            catch E
                rethrow(E)
            end
        end
        
        function q = Get_Position (obj)
            try
                aux = 1;
                while sum(aux)~=0
                    [aux,q] = obj.CoppeliaSim.simxGetJointPosition(obj.ClientID,obj.Joint,obj.CoppeliaSim.simx_opmode_streaming);
                end
            
                q = double(q);
            catch E
                rethrow(E)
            end
        end
        
        function qp = Get_Velocity (obj)
            try
                aux = 1;
                while sum(aux)~=0
                    [aux,qp] = obj.CoppeliaSim.simxGetObjectFloatParameter(obj.ClientID,obj.Joint,2012,obj.CoppeliaSim.simx_opmode_streaming);
                end
            
                qp = double(qp);
            catch E
                rethrow(E)
            end
        end

        function obj = Set_Force (obj,tau)
            try
                obj.CoppeliaSim.simxSetJointTargetVelocity(obj.ClientID,obj.Joint,sign(tau)*1e6,obj.CoppeliaSim.simx_opmode_streaming);
                obj.CoppeliaSim.simxSetJointForce(obj.ClientID,obj.Joint,abs(tau),obj.CoppeliaSim.simx_opmode_streaming);
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