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
        x = x + tasa_aprendizaje * g  # descenso
        puntos.append(x.copy())
        valores.append(funcion(x))

        # imprimir cada iteración
        print(f"{i + 1:<10} {str(x):<25} {funcion(x):<15.8f} {str(g):<25}")

    # imprimir resultado final
    print("\nResultado final:")
    print(f" Punto = {x}")
    print(f" f(x)  = {funcion(x):.8f}")

    return np.array(puntos), np.array(valores)



def graficar(puntos, valores):
    n_dim = puntos.shape[1]

    if n_dim == 1:
        # Caso 1D
        x = np.linspace(-5, 5, 200)
        y = x ** 2
        plt.plot(x, y, label="f(x)")
        plt.plot(puntos[:, 0], valores, "ro-", label="Recorrido")
        plt.title("Gradiente en 1D")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.show()

    elif n_dim == 2:
        # Caso 2D con superficie
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        Z = X ** 2 + Y ** 2

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.6)

        ax.plot(puntos[:, 0], puntos[:, 1], valores, "ro-", label="Recorrido")
        ax.set_title("Gradiente en 2D")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("f(x,y)")
        ax.legend()
        plt.show()

    elif n_dim == 3:
        # Caso 3D (trayectoria en espacio de parámetros)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(puntos[:, 0], puntos[:, 1], puntos[:, 2], "ro-", label="Recorrido")
        ax.set_title("Gradiente en 3D")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
        ax.legend()
        plt.show()

    else:
        # Caso n > 3 → graficar convergencia
        plt.plot(valores, "bo-")
        plt.title(f"Convergencia del Gradiente en {n_dim}D")
        plt.xlabel("Iteración")
        plt.ylabel("f(x)")
        plt.show()

n=3
x0 = np.random.uniform(-5, 5, size=n)  # punto inicial aleatorio en nD

puntos, valores = gradiente_descenso(x0, tasa_aprendizaje=0.1, iteraciones=30)
graficar(puntos, valores)
