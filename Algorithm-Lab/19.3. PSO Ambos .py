import numpy as np
import matplotlib.pyplot as plt
import random

class PSO_G():
    def __init__(self, funcion_objetivo, dimension, num_particulas, minimizar=True,
                 c1=0.5, c2=0.5, Tmax=20, omega=0.8, omega_min=0.3,
                 omega_fijo=False, tipo='gbest', vecinos=2):
        self.funcion_objetivo = funcion_objetivo
        self.dimension = dimension
        self.num_particulas = num_particulas
        self.minimizar = minimizar
        self.c1 = c1
        self.c2 = c2
        self.Tmax = Tmax
        self.omega = omega
        self.omega_min = omega_min
        self.omega_fijo = omega_fijo
        self.tipo = tipo
        self.vecinos = vecinos

    def inicializar(self):
        self.x = np.random.uniform(-5, 5, (self.num_particulas, self.dimension))
        self.v = np.random.uniform(-1, 1, (self.num_particulas, self.dimension))
        self.pbest = np.copy(self.x)
        self.pbest_val = np.array([self.funcion_objetivo(xi) for xi in self.x])

        if self.minimizar:
            self.gbest_index = np.argmin(self.pbest_val)
        else:
            self.gbest_index = np.argmax(self.pbest_val)
        self.gbest = np.copy(self.pbest[self.gbest_index])
        self.gbest_val = self.pbest_val[self.gbest_index]

    def obtener_lbest(self, i):
        indices = [(i + j) % self.num_particulas for j in range(-self.vecinos, self.vecinos + 1)]
        valores = self.pbest_val[indices]

        if self.minimizar:
            idx_mejor = indices[np.argmin(valores)]
        else:
            idx_mejor = indices[np.argmax(valores)]

        return np.copy(self.pbest[idx_mejor])

    def ejecutar(self):
        self.inicializar()
        historial = []

        plt.ion()
        fig, ax = plt.subplots()

        for t in range(self.Tmax):
            if not self.omega_fijo:
                self.omega = self.omega - ((self.omega - self.omega_min) / self.Tmax)

            for i in range(self.num_particulas):
                r1 = random.uniform(0, 1)
                r2 = random.uniform(0, 1)

                if self.tipo == 'gbest':
                    mejor_vecino = self.gbest
                else:
                    mejor_vecino = self.obtener_lbest(i)

                self.v[i] = (self.omega * self.v[i] +
                             self.c1 * r1 * (self.pbest[i] - self.x[i]) +
                             self.c2 * r2 * (mejor_vecino - self.x[i]))
                self.x[i] += self.v[i]

                valor = self.funcion_objetivo(self.x[i])

                if self.minimizar and valor < self.pbest_val[i]:
                    self.pbest[i] = np.copy(self.x[i])
                    self.pbest_val[i] = valor
                elif not self.minimizar and valor > self.pbest_val[i]:
                    self.pbest[i] = np.copy(self.x[i])
                    self.pbest_val[i] = valor

            if self.tipo == 'gbest':
                if self.minimizar:
                    idx = np.argmin(self.pbest_val)
                    if self.pbest_val[idx] < self.gbest_val:
                        self.gbest = np.copy(self.pbest[idx])
                        self.gbest_val = self.pbest_val[idx]
                else:
                    idx = np.argmax(self.pbest_val)
                    if self.pbest_val[idx] > self.gbest_val:
                        self.gbest = np.copy(self.pbest[idx])
                        self.gbest_val = self.pbest_val[idx]

            mejor_actual = np.min(self.pbest_val) if self.minimizar else np.max(self.pbest_val)
            historial.append(mejor_actual)
            print(f"Iter {t+1}/{self.Tmax} - Mejor valor: {mejor_actual:.6f}")

            ax.cla()
            ax.plot(historial, color='blue', linewidth=2)
            ax.set_title(f'Iteración {t + 1}/{self.Tmax} - Mejor valor: {mejor_actual:.6f}')
            ax.set_xlabel('Iteraciones')
            ax.set_ylabel('Mejor valor')
            ax.grid(True)
            plt.pause(0.05)

        plt.ioff()
        plt.show()

        if self.tipo == 'lbest':
            if self.minimizar:
                idx = np.argmin(self.pbest_val)
            else:
                idx = np.argmax(self.pbest_val)
            self.gbest = np.copy(self.pbest[idx])
            self.gbest_val = self.pbest_val[idx]

        return self.gbest, self.gbest_val, historial

def funcion(x):
    x = np.array(x)
    return 1 + np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))

dimension = 5
num_particulas = 20
minimizar = True
c1, c2 = 0.3, 0.5
Tmax = 20
omega = 0.8
omega_fijo=False
tipo="lbest"
vecinos=3


pso = PSO_G(
    funcion_objetivo=funcion,
    dimension=dimension,
    num_particulas=num_particulas,
    minimizar=minimizar,
    c1=c1,
    c2=c2,
    Tmax=Tmax,
    omega=omega,
    omega_fijo=omega_fijo,
    tipo=tipo,
    vecinos=vecinos
)

mejor_sol, mejor_valor, historial = pso.ejecutar()
print("\nMejor solución encontrada:", mejor_sol)
print("Mejor valor de la función:", mejor_valor)
