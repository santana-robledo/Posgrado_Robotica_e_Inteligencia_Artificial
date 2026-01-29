import numpy as np
import matplotlib.pyplot as plt
import random

class Algoritmo_genetico():
    def __init__(self, funcion_objetivo, dimension, limites, num_bits, max_iter, tam_poblacion,
                 prob_cruza, prob_mutacion, minimizar=True, porc_ruleta=0.5, porc_torneo=0.5):
        self.funcion_objetivo = funcion_objetivo
        self.dimension = dimension
        self.limites = limites
        self.num_bits = num_bits
        self.max_iter = max_iter
        self.tam_poblacion = tam_poblacion
        self.prob_cruza = prob_cruza
        self.prob_mutacion = prob_mutacion
        self.minimizar = minimizar
        self.porc_ruleta = porc_ruleta
        self.porc_torneo = porc_torneo

    def evaluar_poblacion(self, poblacion):
        fitness_ = np.array([self.funcion_objetivo(self.decodificar(ind)) for ind in poblacion])
        if self.minimizar:
            fitness = 1 / (1 + fitness_)
        else:
            fitness = fitness_

    def decodificar(self, individuo):
        valores = []
        for i in range(self.dimension):
            bits = individuo[i * self.num_bits:(i + 1) * self.num_bits]
            decimal = int(''.join(str(b) for b in bits), 2)
            valor_real = self.limites[0] + (decimal / (2 ** self.num_bits - 1)) * (self.limites[1] - self.limites[0])
            valores.append(valor_real)
        return np.array(valores)

    def evaluar_poblacion(self, poblacion):
        return np.array([self.funcion_objetivo(self.decodificar(ind)) for ind in poblacion])

    def seleccion_torneo(self, poblacion, fitness, tam_torneo=3):
        seleccionados = []
        for _ in range(len(poblacion)):
            indices = np.random.choice(len(poblacion), tam_torneo, replace=False)
            competidores = fitness[indices]
            if self.minimizar:
                ganador_idx = indices[np.argmin(competidores)]
            else:
                ganador_idx = indices[np.argmax(competidores)]
            seleccionados.append(poblacion[ganador_idx])
        return np.array(seleccionados)

    def seleccion_ruleta(self, poblacion, fitness):
        if self.minimizar:
            inv_fitness = 1 / (1 + fitness)
            prob = inv_fitness / np.sum(inv_fitness)
        else:
            prob = fitness / np.sum(fitness)
        indices = np.random.choice(len(poblacion), size=len(poblacion), p=prob)
        return poblacion[indices]

    def seleccion_padres(self, poblacion, fitness):
        n_ruleta = int(self.tam_poblacion * self.porc_ruleta)
        n_torneo = int(self.tam_poblacion * self.porc_torneo)
        padres_ruleta = self.seleccion_ruleta(poblacion, fitness)[:n_ruleta]
        padres_torneo = self.seleccion_torneo(poblacion, fitness)[:n_torneo]
        return np.vstack((padres_torneo, padres_ruleta))

    def cruzamiento(self, padres):
        hijos = []
        for i in range(0, len(padres), 2):
            padre1, padre2 = padres[i], padres[(i + 1) % len(padres)]
            if random.random() < self.prob_cruza:
                punto = random.randint(1, len(padre1) - 1)
                hijo1 = np.concatenate([padre1[:punto], padre2[punto:]])
                hijo2 = np.concatenate([padre2[:punto], padre1[punto:]])
            else:
                hijo1, hijo2 = padre1.copy(), padre2.copy()
            hijos.append(hijo1)
            hijos.append(hijo2)
        return np.array(hijos[:self.tam_poblacion])

    def mutacion(self, poblacion):
        for i in range(len(poblacion)):
            for j in range(len(poblacion[i])):
                if random.random() < self.prob_mutacion:
                    poblacion[i][j] = 1 - poblacion[i][j]
        return poblacion

    def ejecutar(self):
        poblacion = self.inicializar_poblacion()
        fitness = self.evaluar_poblacion(poblacion)

        mejor_sol = None
        mejor_fit = float('inf') if self.minimizar else -float('inf')
        historial = []

        plt.ion()
        fig, ax = plt.subplots()

        for iter in range(self.max_iter):
            padres = self.seleccion_padres(poblacion, fitness)
            hijos = self.cruzamiento(padres)
            nueva_poblacion = self.mutacion(hijos)
            fitness = self.evaluar_poblacion(nueva_poblacion)
            poblacion = nueva_poblacion

            if self.minimizar:
                idx = np.argmin(fitness)
                if fitness[idx] < mejor_fit:
                    mejor_fit = fitness[idx]
                    mejor_sol = self.decodificar(poblacion[idx])
            else:
                idx = np.argmax(fitness)
                if fitness[idx] > mejor_fit:
                    mejor_fit = fitness[idx]
                    mejor_sol = self.decodificar(poblacion[idx])

            historial.append(mejor_fit)

            print(f"Iteración {iter + 1}: Mejor fitness = {mejor_fit:.6f}, Mejor solución = {mejor_sol}")

            ax.cla()
            ax.plot(historial, color='blue', linewidth=2)
            ax.set_title(f'Iteración {iter + 1}/{self.max_iter}')
            ax.set_xlabel('Iteraciones')
            ax.set_ylabel('Mejor Fitness')
            ax.grid(True)
            plt.pause(0.1)

        plt.ioff()
        plt.show()

        return mejor_sol, mejor_fit, historial

def funcion(x):
    x = np.array(x)
    n = len(x)
    total = 0
    for i in range(1, n + 1):
        suma_parcial = np.sum(x[:i])
        total += suma_parcial ** 2
    return total

ag = Algoritmo_genetico(funcion, dimension=2, limites=[-5, 5],
                        num_bits=10, max_iter=50, tam_poblacion=20,
                        prob_cruza=0.8, prob_mutacion=0.02, minimizar=False)

mejor_sol, mejor_fit, historial = ag.ejecutar()

print("\nMejor solución final encontrada:", mejor_sol)
print("Mejor valor de la función:", mejor_fit)

