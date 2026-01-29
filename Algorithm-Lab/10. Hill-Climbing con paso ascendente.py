import random
import numpy as np
import matplotlib.pyplot as plt


def funcion(x):
    return -np.sum(x**2) + 4*np.sum(x)

def hill_climbing_paso_ascendente(funcion_obj, x_inicial, max_iter=1000, paso=0.5):
    x_actual = np.array(x_inicial)
    mejor_x = x_actual.copy()
    mejor_valor = funcion_obj(x_actual)
    historia = [x_actual.copy()]

    for i in range(max_iter):
        valor_actual=funcion_obj(x_actual)
        mejor_vecino=None
        mejor_valor_vecino=valor_actual

        for dim in range(len(x_actual)):
            x_positivo=x_actual.copy()
            x_positivo[dim]+=paso
            valor_positivo=funcion_obj(x_positivo)

            x_negativo=x_actual.copy()
            x_negativo[dim]-=paso
            valor_negativo=funcion_obj(x_negativo)

            if valor_positivo< mejor_valor_vecino:
                mejor_valor_vecino = valor_positivo
                mejor_vecino=x_positivo

            if valor_negativo< mejor_valor_vecino:
                mejor_valor_vecino=valor_negativo
                mejor_vecino=x_negativo

        if mejor_vecino is not None and mejor_valor_vecino > valor_actual:
            x_actual=mejor_vecino
            mejor_valor=mejor_valor_vecino
            mejor_x=x_actual.copy()
            historia.append(x_actual.copy())
        else:
            break


    return mejor_x, mejor_valor, np.array(historia)


n_dim = 3
x_inicial = np.random.uniform(-10, 10, size=n_dim)
max_iter = 1000
paso = 0.5

x_optimo, valor_optimo, historia = hill_climbing_paso_ascendente(funcion, x_inicial, max_iter, paso)

print(f"Punto inicial: x = {x_inicial}, f(x) = {funcion(x_inicial):.3f}")
print(f"Resultado optimizado: x = {x_optimo}, f(x) = {valor_optimo:.3f}")

if n_dim == 2:
    x1 = np.linspace(-5, 8, 200)
    x2 = np.linspace(-5, 8, 200)
    X1, X2 = np.meshgrid(x1, x2)
    Z = -X1**2 - X2**2 + 4*X1 + 4*X2

    plt.contourf(X1, X2, Z, levels=30, cmap='viridis')
    plt.plot(historia[:, 0], historia[:, 1], 'o--', color='orange', label='Trayectoria')
    plt.scatter(x_optimo[0], x_optimo[1], color='red', label='Extremo encontrado')
    plt.title("Hill Climbing en 2D")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.colorbar(label='f(x1, x2)')
    plt.show()
