import numpy as np
import matplotlib.pyplot as plt
import random

class Evolucion_Diferencial():
    def __init__(self, funcion_objetivo, dimension, limites, max_iter, tam_poblacion,
                 prob_cruza, minimizar=True, factor_escala=0.8):
        self.funcion_objetivo = funcion_objetivo
        self.dimension = dimension
        self.limites = limites
        self.max_iter = max_iter
        self.tam_poblacion = tam_poblacion
        self.prob_cruza = prob_cruza
        self.minimizar = minimizar
        self.factor_escala = factor_escala

    def inicializar_poblacion(self):
        poblacion = np.random.rand(self.tam_poblacion, self.dimension)
        poblacion = self.limites[0] + poblacion * (self.limites[1] - self.limites[0])
        return poblacion

    def mutacion(self, poblacion):
        poblacion_mutante = np.zeros_like(poblacion)
        for i in range(self.tam_poblacion):
            idxs = list(range(self.tam_poblacion))
            idxs.remove(i)
            r1, r2, r3 = random.sample(idxs, 3)
            v_i = poblacion[r1] + self.factor_escala * (poblacion[r2] - poblacion[r3])
            poblacion_mutante[i] = np.clip(v_i, self.limites[0], self.limites[1])
        return poblacion_mutante

    def cruzamiento(self, poblacion, poblacion_mutante):
        poblacion_candidata = np.zeros_like(poblacion)
        for i in range(self.tam_poblacion):
            j_rand = random.randint(0, self.dimension - 1)
            for j in range(self.dimension):
                if random.random() < self.prob_cruza or j == j_rand:
                    poblacion_candidata[i, j] = poblacion_mutante[i, j]
                else:
                    poblacion_candidata[i, j] = poblacion[i, j]
        return poblacion_candidata

    def seleccion(self, poblacion, poblacion_candidata):
        nueva_poblacion = np.zeros_like(poblacion)
        for i in range(self.tam_poblacion):
            f_orig = self.funcion_objetivo(poblacion[i])
            f_cand = self.funcion_objetivo(poblacion_candidata[i])
            if self.minimizar:
                nueva_poblacion[i] = poblacion_candidata[i] if f_cand < f_orig else poblacion[i]
            else:
                nueva_poblacion[i] = poblacion_candidata[i] if f_cand > f_orig else poblacion[i]
        return nueva_poblacion

    def ejecutar(self):
        poblacion = self.inicializar_poblacion()
        historial = []

        if self.minimizar:
            mejor_valor = float("inf")
        else:
            mejor_valor = -float("inf")
        mejor_solucion = None

        # Activar modo interactivo para animación
        plt.ion()
        fig, ax = plt.subplots()

        for iter in range(self.max_iter):
            poblacion_mutante = self.mutacion(poblacion)
            poblacion_candidata = self.cruzamiento(poblacion, poblacion_mutante)
            poblacion = self.seleccion(poblacion, poblacion_candidata)

            # Evaluar la mejor solución en esta generación
            for i in range(self.tam_poblacion):
                val = self.funcion_objetivo(poblacion[i])
                if self.minimizar and val < mejor_valor:
                    mejor_valor = val
                    mejor_solucion = poblacion[i].copy()
                elif not self.minimizar and val > mejor_valor:
                    mejor_valor = val
                    mejor_solucion = poblacion[i].copy()

            historial.append(mejor_valor)

            # Animación de la "L"
            ax.cla()
            ax.plot(historial, color='blue', linewidth=2)
            ax.set_title(f'Iteración {iter+1}/{self.max_iter}')
            ax.set_xlabel('Iteraciones')
            ax.set_ylabel('Mejor valor')
            ax.grid(True)
            plt.pause(0.05)

        plt.ioff()
        plt.show()
        return mejor_solucion, mejor_valor, historial

# --- Función de prueba ---
def esfera(x):
    return np.sum(x**2)

# --- Parámetros ---
dimension = 2
limites = [-5, 5]
tam_poblacion = 20
max_iter = 50
F = 0.8
Cr = 0.7
minimizar = True

ed = Evolucion_Diferencial(
    funcion_objetivo=esfera,
    dimension=dimension,
    limites=limites,
    max_iter=max_iter,
    tam_poblacion=tam_poblacion,
    prob_cruza=Cr,
    minimizar=minimizar,
    factor_escala=F
)

mejor_sol, mejor_valor, historial = ed.ejecutar()
print("Mejor solución encontrada:", mejor_sol)
print("Mejor valor de la función:", mejor_valor)