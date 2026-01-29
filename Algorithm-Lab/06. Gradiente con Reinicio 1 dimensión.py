import matplotlib.pyplot as plt
import numpy as np
import random

# Parámetros iniciales del algoritmo
Rango = [-4, 8]
presicion = 0.0001
paso = 0.2
max_iter = 300
rango_numinicial=[-3,3]
num_intentos=20

# Función objetivo
def funcion(x):
    return np.sin(x) + 0.5 * np.sin(3*x) + 0.3 * np.cos(5*x)

# Derivada
def derivada(x):
    return np.cos(x) + 1.5 * np.cos(3*x) - 1.5 * np.sin(5*x)

# Gradiente descendente
def gradiente_descendente(x_inicial, paso, max_iter):
    puntos_x = []
    puntos_y = []

    x = x_inicial
    puntos_x.append(x)
    puntos_y.append(funcion(x))

    gradiente = derivada(x)

    for i in range(max_iter):
        if abs(gradiente) > presicion:
            gradiente = derivada(x)
            x = x + (paso * gradiente)
        else:
            break
        puntos_x.append(x)
        puntos_y.append(funcion(x))

    return x, gradiente


print(f"{'Iteracion':<12} {'x_inicial':<25}{'X final':<20} {'f(X final)':<20}")
print("-" * 80)

mejores_x = []
mejores_y = []

for i in range(num_intentos):
    valor_inicial = random.uniform(rango_numinicial[0], rango_numinicial[1])
    x_final, gradiente = gradiente_descendente(valor_inicial, paso, max_iter)

    mejores_x.append(x_final)
    mejores_y.append(funcion(x_final))

    print(f"{i + 1:<12} {valor_inicial:<20.3f}{x_final:<20.8f} {funcion(x_final):<20.8f}")
##print(len(mejores_x),mejores_x)
#print(len(mejores_y),mejores_y)
indice_mejor = np.argmax(mejores_y)
mejor_x = mejores_x[indice_mejor]
mejor_y = mejores_y[indice_mejor]

print(f"\n*** MEJOR PUNTO ENCONTRADO ***")
print(f"Intento: #{indice_mejor + 1}")
print(f"Coordenadas: x = {mejor_x:.6f}, f(x) = {mejor_y:.6f}")

# Graficar
x = np.linspace(-10, 10, 400)
y_func = funcion(x)

plt.plot(x, y_func, label="f(x)", linewidth=2)
for i in range(len(mejores_x)):
    if i == 0:  # Solo agregar leyenda para el primer punto
        plt.plot(mejores_x[i], mejores_y[i], 'go', label='Puntos')
    else:  # Los demás puntos sin leyenda
        plt.plot(mejores_x[i], mejores_y[i], 'go', label='')

plt.plot(mejor_x, mejor_y, 'ro', markersize=10, label=f'Mejor punto: f(x) = {mejor_y:.4f}')

plt.title("Gradiente con Reinicio", fontsize=14)
plt.xlabel("X", fontsize=12)
plt.ylabel("f(X)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
