import numpy as np
import matplotlib.pyplot as plt
import math

x_actual=2.0
f_actual=4.0
rango=[-10,10]
Δf=2.25
T_k=5

def funcion(x):
    return 50 * x * np.exp(-0.01 * x)

def generar_vecino(x_actual,rango,paso=0.5):
    desfase = np.random.normal(0,paso)
    x_vecino=x_actual+desfase
    x_vecino = max(rango[0],min(rango[1],x_vecino))
    return x_vecino

def probabilidad_aceptacion(f_actual,f_vecino,temperatura):
    if f_vecino>f_actual:
        return 1
    else:
        return math.exp((f_vecino - f_actual) / temperatura)

def simulated_annealing(funcion_objetivo,rango,max_iter=1000):
    x_actual=np.random.uniform(rango[0],rango[1])
    f_actual=funcion_objetivo(x_actual)

    historia=[]
    T0 = 100  # Temperatura inicial
    alpha = 0.95  # Tasa de enfriamiento
    mejor_x = x_actual
    mejor_f = f_actual

    for i in range(max_iter):
        T=T0*(alpha ** i)

        x_vecino=generar_vecino(x_actual,rango)
        f_vecino=funcion_objetivo(x_vecino)

        prob_accept= probabilidad_aceptacion(f_actual,f_vecino,T)

        if np.random.random()< prob_accept:
            x_actual,f_actual=x_vecino, f_vecino
            aceptado=True
        else:
            aceptado = False

        if f_actual < mejor_f:
            mejor_x, mejor_f = x_actual, f_actual

        historia.append((x_actual,f_actual))
        print(f"{i:4d} | {x_actual:11.4f} | {f_actual:11.4f} | {T:7.2f} | {prob_accept:10.4f} | {aceptado:8} | {mejor_f:8.4f}")


    return x_actual,f_actual,historia


print("Iter |    x_actual | f(x_actual) |   Temp  | Prob_Acept | Aceptado | Mejor_f")
print("-" * 75)
x_actual,f_actual,historia=simulated_annealing(funcion,rango,max_iter=1000)
print(f"Minimo en {x_actual} con valor de {f_actual}")

x = np.linspace(rango[0]-100, rango[1]+100, 400)
y = funcion(x)
plt.plot(x, y, label="f(x)")
plt.scatter(x_actual, f_actual, color="green", label=f'Punto encontrado ≈ ({x_actual:.2f}, {f_actual:.2f})',alpha=1,zorder=5,s=100)
x_hist=[h[0] for h in historia]
f_hist=[h[1] for h in historia]
plt.plot(x_hist,f_hist,"ro-", label="Trayectoria",alpha=0.7)
plt.title("Gráfica de Función y Mínimo Encontrado")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()



