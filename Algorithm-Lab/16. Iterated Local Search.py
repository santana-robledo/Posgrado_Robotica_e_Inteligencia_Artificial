import numpy as np
import matplotlib.pyplot as plt
import math


def funcion(x):
    return -(x-3)**2 + 10


def generar_vecino(x_actual, rango, paso=0.1):
    perturbacion = np.random.normal(0, paso)
    x_vecino = x_actual + perturbacion
    x_vecino = max(rango[0], min(rango[1], x_vecino))
    return x_vecino


def busqueda_local(x_inicial, funcion, rango, max_iter_local=100):
    x_actual = x_inicial
    f_actual = funcion(x_actual)

    trayectoria_local_x = [x_actual]
    trayectoria_local_f = [f_actual]

    for i in range(max_iter_local):
        x_vecino = generar_vecino(x_actual, rango, paso=0.1)
        f_vecino = funcion(x_vecino)

        if f_vecino < f_actual:
            x_actual, f_actual = x_vecino, f_vecino
            trayectoria_local_x.append(x_actual)
            trayectoria_local_f.append(f_actual)

    return x_actual, f_actual, trayectoria_local_x, trayectoria_local_f


def perturbacion(solucion, rango, fuerza=2.0):
    perturbacion = np.random.normal(0, fuerza)
    x_perturbado = solucion + perturbacion
    x_perturbado = max(rango[0], min(rango[1], x_perturbado))
    return x_perturbado


def criterio_aceptacion(f_nuevo, f_actual, mejor_f, umbral=0.1):
    if f_nuevo < f_actual:
        return True
    elif f_nuevo < f_actual + umbral:
        return np.random.random() < 0.3
    else:
        return False


def iterated_local_search(funcion_objetivo, rango, max_iter_ILS=50):
    historial_global = []
    trayectoria_completa_x = []
    trayectoria_completa_f = []

    x_actual = np.random.uniform(rango[0], rango[1])
    f_actual = funcion_objetivo(x_actual)

    trayectoria_completa_x.append(x_actual)
    trayectoria_completa_f.append(f_actual)

    x_actual, f_actual, trayectoria_local_x, trayectoria_local_f = busqueda_local(x_actual, funcion_objetivo, rango)
    mejor_x, mejor_f = x_actual, f_actual

    trayectoria_completa_x.extend(trayectoria_local_x[1:])
    trayectoria_completa_f.extend(trayectoria_local_f[1:])

    print("Iter_ILS |    x_actual | f_actual | Mejor_f | Aceptado")
    print("-" * 60)

    for iter_ILS in range(max_iter_ILS):
        x_perturbado = perturbacion(x_actual, rango, fuerza=2.0)
        x_nuevo, f_nuevo, trayectoria_local_x, trayectoria_local_f = busqueda_local(x_perturbado, funcion_objetivo,
                                                                                    rango)
        aceptado = criterio_aceptacion(f_nuevo, f_actual, mejor_f)

        if aceptado:
            x_actual, f_actual = x_nuevo, f_nuevo

        if f_actual < mejor_f:
            mejor_x, mejor_f = x_actual, f_actual

        trayectoria_completa_x.extend(trayectoria_local_x)
        trayectoria_completa_f.extend(trayectoria_local_f)

        historial_global.append((x_actual, f_actual, mejor_f, aceptado))

        print(f"{iter_ILS:8d} | {x_actual:11.4f} | {f_actual:8.4f} | {mejor_f:8.4f} | {aceptado:8}")

    return mejor_x, mejor_f, historial_global, trayectoria_completa_x, trayectoria_completa_f


if __name__ == "__main__":
    rango = [-10, 10]
    max_iter_ILS = 30

    x_opt, f_opt, historial, trayectoria_x, trayectoria_f = iterated_local_search(funcion, rango, max_iter_ILS)

    print(f"\nMínimo encontrado: x = {x_opt:.6f}, f(x) = {f_opt:.6f}")

    x = np.linspace(rango[0], rango[1], 400)
    y = funcion(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2, alpha=0.7, label='f(x) = x²')
    plt.plot(trayectoria_x, trayectoria_f, 'ro-', markersize=2, alpha=0.6, linewidth=0.5, label='Trayectoria ILS')

    for i, (x_val, f_val) in enumerate(zip(trayectoria_x, trayectoria_f)):
        plt.scatter(x_val, f_val, color='red', s=10, alpha=0.4)
        if i % 50 == 0:
            plt.annotate(f'{i}', (x_val, f_val), xytext=(5, 5),
                         textcoords='offset points', fontsize=6, alpha=0.7)

    plt.scatter(x_opt, f_opt, color='green', s=100, marker='*',
                label=f'Mínimo: ({x_opt:.4f}, {f_opt:.4f})', zorder=5)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Iterated Local Search - Trayectoria Punto por Punto")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.show()
