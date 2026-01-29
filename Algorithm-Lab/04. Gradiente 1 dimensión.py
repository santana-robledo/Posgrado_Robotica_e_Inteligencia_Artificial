import matplotlib.pyplot as plt
import numpy as np

# Parámetros iniciales del algoritmo
Rango = [-4, 8]  # Rango de visualización (no utilizado en el algoritmo)
presicion = 0.0001  # Precisión para el criterio de parada
x_inicial = 3  # Punto inicial para comenzar la optimización
paso = 0.2  # Tasa de aprendizaje (learning rate)
max_iter = 300  # Número máximo de iteraciones permitidas

# Listas para almacenar el historial de la optimización
puntos_x = []  # Almacenará los valores de x en cada iteración
puntos_y = []  # Almacenará los valores de f(x) en cada iteración
iteraciones = [0]  # Almacenará el número de iteración


# Función objetivo que queremos minimizar: f(x) = x² + 1
def funcion(x):
    return np.sin(x)


# Derivada de la función objetivo: f'(x) = 2x
def derivada(x):
    return np.cos(x)


# Implementación del algoritmo de descenso de gradiente
def gradiente_descendente(x_inicial, paso, max_iter):
    x = x_inicial  # Inicializar x con el valor inicial
    puntos_x.append(x)  # Guardar el punto inicial
    puntos_y.append(funcion(x))  # Guardar el valor de la función en el punto inicial

    gradiente = derivada(x)  # Calcular el gradiente en el punto inicial

    # Encabezado de la tabla de resultados
    print(f"{'Iteracion':<12} {'X':<20} {'f(x)':<25} {'Gradiente':<20}")
    print("-" * 80)

    # Bucle principal de optimización
    for i in range(max_iter):
        if abs(gradiente) > presicion:#Usamos abs para que el mismo algoritmo funciones con gradiente ascendente y descendente
            gradiente = derivada(x)  # Calcular el gradiente en el punto actual
            x = x - (paso * gradiente)  # Actualizar x: movimiento en dirección opuesta al gradiente

            # Almacenar historial
            puntos_x.append(x)
            puntos_y.append(funcion(x))
            iteraciones.append(i + 1)  # +1 porque i comienza en 0

            # Mostrar información de la iteración actual
            print(f"{i + 1:<12} {x:<20.8f} {funcion(x):<25.8f} {gradiente:<20.8f}")
        else:
            break  # Salir del bucle si se alcanza la convergencia

    return puntos_x, puntos_y, iteraciones


# Ejecutar el algoritmo de descenso de gradiente
gradiente_descendente(x_inicial, paso, max_iter)

# Preparar datos para visualización
x = np.linspace(-10, 10, 400)  # Crear 400 puntos en el rango [-10, 10]
y_func = funcion(x)  # Calcular f(x) para cada punto

# Crear visualización gráfica
plt.plot(x, y_func, label="Función f(x) = x² + 1", linewidth=2)  # Graficar la función
plt.plot(x_inicial, funcion(x_inicial), 'go', markersize=8, label="Punto Inicial")  # Punto inicial
plt.plot(puntos_x, puntos_y, "ro-", label="Trayectoria", markersize=4, linewidth=1)  # Trayectoria
plt.plot(puntos_x[-1], puntos_y[-1], "bo", markersize=8, label="Punto Final")  # Punto final

# Configuración del gráfico
plt.title("Algoritmo de Gradiente", fontsize=14)
plt.xlabel("X", fontsize=12)
plt.ylabel("f(X)", fontsize=12)
plt.xlim(-5, 5)
plt.ylim(-2, 5)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
