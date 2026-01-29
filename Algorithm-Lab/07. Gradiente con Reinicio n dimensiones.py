import numpy as np
import matplotlib.pyplot as plt

def funcion(x):
    return np.sum(x ** 2)

def gradiente(x):
    return 2 * x

def gradiente_descenso(x_inicial, tasa_aprendizaje=0.1, iteraciones=30):
    puntos = [x_inicial]
    valores = [funcion(x_inicial)]
    x = x_inicial.copy()

    print(f"{'Iteración':<10} {'x':<25} {'f(x)':<15} {'gradiente':<25}")
    print("-" * 80)

    for i in range(iteraciones):
        g = gradiente(x)
        x = x + tasa_aprendizaje * g
        puntos.append(x.copy())
        valores.append(funcion(x))

        # imprimir cada iteración
        print(f"{i + 1:<10} {str(x):<25} {funcion(x):<15.8f} {str(g):<25}")

    return np.array(puntos), np.array(valores)

def graficar(mejores_x, mejores_y, mejor_x, mejor_y, funcion):
    n_dim = np.array(mejores_x).shape[1]

    if n_dim == 1:
        # --- Caso 1D ---
        x = np.linspace(-10, 10, 400)
        y_func = [funcion(np.array([xi])) for xi in x]
        plt.plot(x, y_func, label="f(x)", linewidth=2)

        for i in range(len(mejores_x)):
            if i == 0:
                plt.plot(mejores_x[i], mejores_y[i], 'go', label='Puntos')
            else:
                plt.plot(mejores_x[i], mejores_y[i], 'go')

        plt.plot(mejor_x, mejor_y, 'ro', markersize=10,
                 label=f'Mejor punto: f(x) = {mejor_y:.4f}')
        plt.title("Gradiente con Reinicio (1D)", fontsize=14)
        plt.xlabel("X", fontsize=12)
        plt.ylabel("f(X)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

    elif n_dim == 2:
        # --- Caso 2D ---
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = funcion(np.array([X[i, j], Y[i, j]]))

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.6)

        for i in range(len(mejores_x)):
            ax.scatter(mejores_x[i][0], mejores_x[i][1], mejores_y[i],
                       c='g', marker='o')

        ax.scatter(mejor_x[0], mejor_x[1], mejor_y,
                   c='r', marker='o', s=100, label=f"Mejor f(x)={mejor_y:.4f}")

        ax.set_title("Gradiente con Reinicio (2D)")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("f(x1,x2)")
        ax.legend()
        plt.show()

    elif n_dim == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        for i in range(len(mejores_x)):
            ax.scatter(mejores_x[i][0], mejores_x[i][1], mejores_x[i][2],
                       c='g', marker='o')

        ax.scatter(mejor_x[0], mejor_x[1], mejor_x[2],
                   c='r', marker='o', s=100, label="Mejor punto")

        ax.set_title("Gradiente con Reinicio (3D)")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
        ax.legend()
        plt.show()

    else:
        # --- Caso n > 3: solo convergencia ---
        plt.plot(mejores_y, "bo-")
        plt.title(f"Convergencia del Gradiente en {n_dim}D")
        plt.xlabel("Intento")
        plt.ylabel("f(x)")
        plt.grid(True, alpha=0.3)
        plt.show()

n = 3
num_intentos = 20
print(f"{'Iteracion':<12} {'x_inicial':<25}{'X final':<20} {'f(X final)':<20}")
print("-" * 80)

mejores_x = []
mejores_y = []

for i in range(num_intentos):
    valor_inicial = np.random.uniform(-5, 5, size=n)
    trayectoria, valores = gradiente_descenso(valor_inicial, tasa_aprendizaje=0.1, iteraciones=30)

    x_final = trayectoria[-1]
    y_final = valores[-1]

    mejores_x.append(x_final)
    mejores_y.append(y_final)

    print(f"{i + 1:<12} {valor_inicial} {x_final} {y_final:.8f}")


indice_mejor = np.argmax(mejores_y)
mejor_x = mejores_x[indice_mejor]
mejor_y = mejores_y[indice_mejor]

print(f"\n*** MEJOR PUNTO FINAL ENCONTRADO ***")
print(f"Intento: #{indice_mejor + 1}")
print(f"Coordenadas: x = {mejor_x}, f(x) = {mejor_y:.6f}")

graficar(mejores_x, mejores_y, mejor_x, mejor_y, funcion)

