function Hx = coupling_function(x, nonlinear_coupling)

if nonlinear_coupling
    Hx = [tanh(x(1));
          x(2);
          0.5*sin(x(3))];
else
    Hx = x;
end

end