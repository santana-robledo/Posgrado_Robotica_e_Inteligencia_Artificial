clear all
close all
clc

S = 3;
dt = 0.01;

pendulo = Bot_Pendulo_Doble(dt);

for t=dt:dt:S
    q = pendulo.Get_Position();
    qp = pendulo.Get_Velocity();
    
    u = [0 0]';

    pendulo.Set_Force(u);
    pendulo.Simulation_Step();
end

pendulo.Stop_Simulation();
