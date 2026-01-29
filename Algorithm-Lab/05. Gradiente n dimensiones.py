##https://www.youtube.com/watch?v=gsfbWn4Gy5Q&t=1542s

import numpy as np
import matplotlib.pyplot as plt

presicion = 0.0001  # Precisión para el criterio de parada


# Define la función objetivo: f(x,y) = sin(5x)*cos(5y)/5
def funcion(x, y):
    return np.sin(5 * x) * np.cos(5 * y) / 5


# Calcula el gradiente de la función: vector de derivadas parciales
def gradiente(x, y):
    # ∂f/∂x = cos(5x)*cos(5y), ∂f/∂y = -sin(5x)*sin(5y)
    return np.cos(5 * x) * np.cos(5 * y), -np.sin(5 * x) * np.sin(5 * y)


# Crea mallas de coordenadas para visualización
x = np.arange(-1, 1, 0.05)  # Valores de x desde -1 hasta 1
y = np.arange(-1, 1, 0.05)  # Valores de y desde -1 hasta 1

X, Y = np.meshgrid(x, y)  # Crea matrices de coordenadas
Z = funcion(X, Y)  # Evalúa la función en todas las coordenadas

# Configuración inicial
posicion = (0.7, 0.4, funcion(0.7, 0.4))  # Punto inicial (x, y, z)
paso = 0.01  # Tasa de aprendizaje (learning rate)
ax = plt.subplot(projection="3d", computed_zorder=False)

# Encabezado de la tabla con información del gradiente
print(f"{'Iteracion':<12} {'X':<15} {'Y':<15} {'f(x,y)':<15} {'Gradiente X':<15} {'Gradiente Y':<15} {'Magnitud':<15}")
print("-" * 100)

# Bucle de optimización (1000 iteraciones)
for i in range(1000):
    # Calcula el gradiente en la posición actual
    x_derivada, y_derivada = gradiente(posicion[0], posicion[1])

    # Calcula la magnitud del vector gradiente
    magnitud_gradiente = np.sqrt(x_derivada ** 2 + y_derivada ** 2)

    # Verifica si la magnitud del gradiente es menor que la precisión
    if magnitud_gradiente > presicion:
        # Actualiza la posición: sigue la dirección opuesta al gradiente
        x_new = posicion[0] + paso * x_derivada
        y_new = posicion[1] + paso * y_derivada
        z_new = funcion(x_new, y_new)
        posicion = (x_new, y_new, z_new)

        # Imprime los valores de la iteración actual, incluyendo ambos gradientes
        print(
            f"{i + 1:<12} {x_new:<15.8f} {y_new:<15.8f} {z_new:<15.8f} {x_derivada:<15.8f} {y_derivada:<15.8f} {magnitud_gradiente:<15.8f}")

        # Visualización
        ax.plot_surface(X, Y, Z, cmap="Blues", zorder=0, alpha=0.6)  # Superficie de la función
        ax.scatter(posicion[0], posicion[1], posicion[2],
                   color="magenta", s=50, zorder=1)  # Punto actual


        plt.pause(0.001)  # Pausa breve para animación
        ax.clear()  # Limpia el gráfico para el siguiente frame
    else:
        print(f"\nConvergencia alcanzada en la iteración {i + 1}")
        print(f"Magnitud del gradiente: {magnitud_gradiente:.8f} < {presicion}")
        break  # Salir del bucle si se alcanza la convergencia

# Si se completaron todas las iteraciones sin converger
if i == 999:
    print(f"\nMáximo de iteraciones alcanzado ({i + 1}) sin converger completamente")
    print(f"Magnitud final del gradiente: {magnitud_gradiente:.8f}")

plt.show()
