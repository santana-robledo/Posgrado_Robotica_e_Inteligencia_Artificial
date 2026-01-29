import numpy as np
import matplotlib.pyplot as plt

def firefly(funcion_objetivo, tamaño_poblacion, dimension, limites, iteraciones,
            alpha, beta, gamma, minimizar=True):

    lim_inf, lim_sup = limites# Límites inferior y superior del espacio de búsqueda

    # Población inicial de luciérnagas, posiciones aleatorias
    fireflies = np.random.uniform(lim_inf, lim_sup, (tamaño_poblacion, dimension))

    # Calcular fitness
    fitness = np.array([funcion_objetivo(x) for x in fireflies])

    historial = []

    for t in range(iteraciones):

        for i in range(tamaño_poblacion):
            for j in range(tamaño_poblacion):

                # Adaptación correcta  de la función para minimización o maximización
                if minimizar:
                    condicion = fitness[j] < fitness[i]
                else:
                    condicion = fitness[j] > fitness[i]

                # Si j es más brillante/mejor que i → i se mueve hacia j
                if condicion:

                    r = np.linalg.norm(fireflies[i] - fireflies[j])# Distancia entre luciérnagas
                    beta_at = beta * np.exp(-gamma * r**2)# Atracción según distancia

                    # Calculamos Movimiento
                    fireflies[i] = (
                        fireflies[i]
                        + beta_at * (fireflies[j] - fireflies[i])
                        + alpha * (np.random.rand(dimension) - 0.5)
                    )

                    # Mantener dentro de límites
                    fireflies[i] = np.clip(fireflies[i], lim_inf, lim_sup)

                    # Recalcular fitness despues del movimiento
                    fitness[i] = funcion_objetivo(fireflies[i])

        # Mejor valor de la iteración
        best_val = np.min(fitness) if minimizar else np.max(fitness)
        historial.append(best_val)

        print(f"Iteración {t+1}/{iteraciones} — Mejor valor: {best_val:.6f}")

    # Seleccionar mejor luciérnaga final
    best_index = np.argmin(fitness) if minimizar else np.argmax(fitness)

    plt.figure(figsize=(7,4))
    plt.plot(historial, linewidth=2)
    plt.title("Progreso del mejor valor por iteración")
    plt.xlabel("Iteraciones")
    plt.ylabel("Mejor valor")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return fireflies[best_index]

def funcion(x):
    x = np.array(x)
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

######   Parametros  ######
funcion_objetivo = funcion
tamaño_poblacion = 20
dimension = 2
limites = (-5,5)
iteraciones = 40
alpha = 0.2
beta = 1.0
gamma = 1.0
minimizar = False

mejor_sol = firefly(
    funcion_objetivo=funcion_objetivo,
    tamaño_poblacion=tamaño_poblacion,
    dimension=dimension,
    limites=limites,
    iteraciones=iteraciones,
    alpha=alpha,
    beta=beta,
    gamma=gamma,
    minimizar=minimizar
)

print("\nMejor solución:", mejor_sol)
