import numpy as np
import matplotlib.pyplot as plt

# ---------------- FUNCIONES ----------------

def esfera(x):
    x = np.array(x)
    return np.sum(x ** 2)

def quadrico(x):
    x = np.array(x)
    n = len(x)
    total = 0
    for i in range(1, n + 1):
        total += np.sum(x[:i]) ** 2
    return total

def ackley(x):
    x = np.array(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.mean(x ** 2))) - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e

def bohachevsky(x):
    x1, x2 = x
    return x1 ** 2 + 2 * x2 ** 2 - 0.3 * np.cos(3 * np.pi * x1) - 0.4 * np.cos(4 * np.pi * x2) + 0.7

def colville(x):
    x = np.pad(np.array(x), (0, max(0, 4 - len(x))), constant_values=1)
    x1, x2, x3, x4 = x[:4]
    return 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2 + 90 * (x4 - x3 ** 2) ** 2 + (1 - x3) ** 2 + \
        10.1 * ((x2 - 1) ** 2 + (x4 - 1) ** 2) + 19.8 * (x2 - 1) * (x4 - 1)

def easom(x):
    x1, x2 = x
    return -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi) ** 2 - (x2 - np.pi) ** 2)

def griewank(x):
    x = np.array(x)
    return 1 + np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))

def hyperellipsoid(x):
    x = np.array(x)
    return np.sum((np.arange(1, len(x) + 1) ** 2) * x ** 2)

def rastrigin(x):
    x = np.array(x)
    return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10)

def rosenbrock(x):
    x = np.array(x)
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

def schwefel(x):
    x = np.array(x)
    return np.sum(x * np.sin(np.sqrt(np.abs(x))) + 418.9829)

# ---------------- GRAFICAR FUNCIONES ----------------

def graficar_funcion_2d(func, title, x_range=(-5, 5), y_range=(-5, 5)):
    x = np.linspace(x_range[0], x_range[1], 200)
    y = np.linspace(y_range[0], y_range[1], 200)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[func([x1, x2]) for x1, x2 in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

    fig = plt.figure(figsize=(12, 5))

    # -------- VISTA DE CONTORNO --------
    ax1 = fig.add_subplot(1, 2, 1)
    cont = ax1.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(cont, ax=ax1, label='f(x1,x2)')
    ax1.set_title(f'{title} - Vista Contorno')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')

    # -------- VISTA 3D --------
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax2.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    fig.colorbar(surf, ax=ax2, label='f(x1,x2)')
    ax2.set_title(f'{title} - Vista 3D')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_zlabel('f(x1,x2)')

    plt.tight_layout()
    plt.show()


def evaluar_n_dim(func, n_dim, rango=(-5, 5)):
    punto = np.random.uniform(rango[0], rango[1], n_dim)
    resultado = func(punto)
    print(f"\nFunción: {func.__name__.capitalize()} ({n_dim}D)")
    print(f"Punto evaluado: {punto}")
    print(f"Resultado: {resultado}")

    # ---- Simulación de convergencia ----
    iteraciones = np.arange(1, 51)
    valores = [func(np.random.uniform(rango[0]/i, rango[1]/i, n_dim)) for i in iteraciones]

    plt.figure(figsize=(7,4))
    plt.plot(iteraciones, valores, marker='o')
    plt.title(f"Convergencia simulada - {func.__name__.capitalize()} {n_dim}D")
    plt.xlabel("Iteración")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.show()

# ---------------- MENU ----------------

funciones = {
    "1": ("Sphere", esfera, (-5, 5)),
    "2": ("Quadric", quadrico, (-2, 2)),
    "3": ("Ackley", ackley, (-5, 5)),
    "4": ("Bohachevsky", bohachevsky, (-1, 1)),
    "5": ("Colville", colville, (-2, 2)),
    "6": ("Easom", easom, (0, 2 * np.pi)),
    "7": ("Griewank", griewank, (-10, 10)),
    "8": ("Hyperellipsoid", hyperellipsoid, (-1, 1)),
    "9": ("Rastrigin", rastrigin, (-5.12, 5.12)),
    "10": ("Rosenbrock", rosenbrock, (-2, 2)),
    "11": ("Schwefel", schwefel, (-500, 500))
}

while True:
    print("""
    ¿Qué función desea probar?
    1. Esfera
    2. Quadric
    3. Ackley
    4. Bohachevsky
    5. Colville
    6. Easom
    7. Griewank
    8. Hyperellipsoid
    9. Rastrigin
    10. Rosenbrock
    11. Schwefel
    0. Salir
    """)
    opcion = input("Selecciona una opción: ").strip()
    if opcion == "0":
        break

    if opcion in funciones:
        nombre, funcion, rango = funciones[opcion]
        n_dim = int(input(f"¿Cuántas dimensiones quieres evaluar para {nombre}? "))
        if n_dim == 2:
            graficar_funcion_2d(funcion, nombre, rango, rango)
        else:
            evaluar_n_dim(funcion, n_dim, rango)
    else:
        print("Opción no válida.")

    seguir = input("\n¿Deseas probar otra función? (sí/no): ").strip().lower()
    if seguir in ["no", "n"]:
        break
