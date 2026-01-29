import pygame
import chess
import random
import numpy as np
import matplotlib.pyplot as plt

class Chess():
    def __init__(self, max_iter, tam_poblacion, prob_cruza, prob_mutacion, porc_ruleta=0.5, porc_torneo=0.5,num_piezas=3,peso_cobertura=0.5,peso_valor_piezas=0.3,peso_penalizacion=0.2):
        self.logic_board = chess.Board.empty()
        self.max_iter = max_iter
        self.tam_poblacion = tam_poblacion
        self.prob_cruza = prob_cruza
        self.prob_mutacion = prob_mutacion
        self.porc_ruleta = porc_ruleta
        self.porc_torneo = porc_torneo
        self.num_piezas=num_piezas
        self.peso_cobertura = peso_cobertura
        self.peso_valor_piezas = peso_valor_piezas
        self.peso_penalizacion = peso_penalizacion

        pygame.init()
        self.WIDTH, self.HEIGHT = 640, 640
        self.SQUARE_SIZE = self.WIDTH // 8
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.DOUBLEBUF)
        pygame.display.set_caption("Algoritmo Genético - Ajedrez")

        self.LIGHT = (240, 217, 181) #Marrón Claro
        self.DARK = (181, 136, 99) #Marrón
        self.COLOR_COBERTURA=(0,255,0,60) #Verde claro
        self.stats_font = pygame.font.SysFont("Arial", 20) #Tipografía tablero
        self.font = pygame.font.SysFont("Arial", 48) #Tipografía fichas

        self.tablero_surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.dibujar_tablero_en_memoria()

        self.piezas_disponibles = [chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.QUEEN]
        self.limites_piezas = { chess.QUEEN: 1,chess.ROOK: 2,chess.BISHOP: 2,chess.KNIGHT: 3 }
        self.valor_piezas = { chess.KNIGHT: 3,chess.BISHOP: 3, chess.ROOK: 5,chess.QUEEN: 9, }
        self.PIECE_UNICODE = { chess.QUEEN: "Q",chess.ROOK: "R",chess.BISHOP: "B",chess.KNIGHT: "N"}

        #DIMENSIONES PARA PANEL
        self.WIDTH, self.HEIGHT = 800, 640  # Más ancho para el panel
        self.SQUARE_SIZE = 80
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.DOUBLEBUF)
        self.mejor_fitness_global = 0
        self.mejor_iteracion_global = 0
        self.casillas_controladas_global = 0

        if not self.verificar_parametros():
            print("Corrige los parámetros antes de continuar.")
            return  # Detener la ejecución si hay errores

        self.loop()
    def dibujar_tablero_en_memoria(self):##Crear tablero y guardar en memoria
        for row in range(8):
            for col in range(8):
                color = self.LIGHT if (row + col) % 2 == 0 else self.DARK
                pygame.draw.rect( self.tablero_surface,color,(col * self.SQUARE_SIZE, row * self.SQUARE_SIZE, self.SQUARE_SIZE, self.SQUARE_SIZE))
    def poner_pieza(self, pieza, fila, columna):##Pone pieza en tablero
        square = chess.square(columna, fila)
        self.logic_board.set_piece_at(square, chess.Piece(pieza, chess.WHITE))
    def dibujar_pieza(self, pieza, fila, columna):
        simbolo = self.PIECE_UNICODE[pieza]
        text = self.font.render(simbolo, True, (0, 0, 0))
        x = columna * self.SQUARE_SIZE + 20
        y = fila * self.SQUARE_SIZE + 10
        self.screen.blit(text, (x, y))
    def calcular_cobertura(self):##Calcula cobertura de una pieza
        cobertura = set()
        for square, piece in self.logic_board.piece_map().items():
            ataques = self.logic_board.attacks(square)
            cobertura.update(ataques)#set con todas las casillas que están siendo cubiertas por cualquier pieza del tablero
        return cobertura
    def resaltar_cobertura(self):#Pinta las casillas que estan siendo atacadas
        surface = pygame.Surface((self.SQUARE_SIZE, self.SQUARE_SIZE), pygame.SRCALPHA)
        surface.fill(self.COLOR_COBERTURA)
        cobertura = self.calcular_cobertura()
        for square in cobertura:
            fila = chess.square_rank(square)
            columna = chess.square_file(square)
            self.screen.blit(surface, (columna * self.SQUARE_SIZE, fila * self.SQUARE_SIZE))

    def render(self):
        self.screen.fill((0, 0, 0))
        self.screen.blit(self.tablero_surface, (0, 0))
        self.resaltar_cobertura()
        for square, piece in self.logic_board.piece_map().items():
            fila = chess.square_rank(square)
            columna = chess.square_file(square)
            self.dibujar_pieza(piece.piece_type, fila, columna)

        # DIBUJAR COORDENADAS DEL TABLERO
        letra_font = pygame.font.SysFont("Arial", 11, bold=True)
        letras = "ABCDEFGH"
        for col in range(8):
            text = letra_font.render(letras[col], True, (0, 0, 0))
            x = col * self.SQUARE_SIZE + self.SQUARE_SIZE // 2 - 5
            y = 8 * self.SQUARE_SIZE + 5
            self.screen.blit(text, (x, y))

        for fila in range(8):
            num = str(8 - fila)
            text = letra_font.render(num, True, (0, 0, 0))
            x = 8 * self.SQUARE_SIZE + 5
            y = fila * self.SQUARE_SIZE + self.SQUARE_SIZE // 2 - 10
            self.screen.blit(text, (x, y))

        # DIBUJAR PANEL DE ESTADÍSTICAS
        self.dibujar_panel_estadisticas()

        pygame.display.flip()

    def dibujar_panel_estadisticas(self):
        """Dibuja el panel de estadísticas al lado del tablero"""
        panel_x = 640  # Empieza después del tablero (8*80=640)
        panel_width = 160

        # Fondo del panel
        pygame.draw.rect(self.screen, (240, 240, 240), (panel_x, 0, panel_width, self.HEIGHT))
        pygame.draw.rect(self.screen, (100, 100, 100), (panel_x, 0, panel_width, self.HEIGHT), 2)

        # Título
        title_font = pygame.font.SysFont("Arial", 18, bold=True)
        title = title_font.render("MEJOR SOLUCIÓN", True, (0, 0, 0))
        self.screen.blit(title, (panel_x + 20, 20))

        # Estadísticas
        stats_font = pygame.font.SysFont("Arial", 16)
        stats = [
            f"Fitness: {self.mejor_fitness_global:.2f}",
            f"Iteración: {self.mejor_iteracion_global}",
            f"Casillas: {self.casillas_controladas_global}/64",
            f"Piezas: {self.num_piezas}",
            f"",
            f"Población: {self.tam_poblacion}",
            f"Iteraciones: {self.max_iter}",
        ]

        # Dibujar estadísticas
        for i, stat in enumerate(stats):
            text = stats_font.render(stat, True, (0, 0, 0))
            self.screen.blit(text, (panel_x + 10, 60 + i * 25))

        ##AQUI EMPIEZA EL ALGORITMO
    def verificar_parametros(self):
        errores = []
        suma_pesos = self.peso_cobertura + self.peso_valor_piezas + self.peso_penalizacion
        if abs(suma_pesos - 1.0) > 0.001:  # Usamos tolerancia para floats
            errores.append(f"Los pesos deben sumar 1.0 (actual: {suma_pesos:.3f})")

        suma_seleccion = self.porc_ruleta + self.porc_torneo
        if abs(suma_seleccion - 1.0) > 0.001:
            errores.append(f"Los porcentajes de selección deben sumar 1.0 (actual: {suma_seleccion:.3f})")

        if not (0 <= self.prob_cruza <= 1):
            errores.append(f"Probabilidad de cruza debe estar entre 0 y 1 (actual: {self.prob_cruza})")

        if not (0 <= self.prob_mutacion <= 1):
            errores.append(f"Probabilidad de mutación debe estar entre 0 y 1 (actual: {self.prob_mutacion})")

        if self.num_piezas > 64:
            errores.append(f"¡ADVERTENCIA! Número de piezas ({self.num_piezas}) excede el máximo (64).")

        if errores:
            print("ERRORES EN PARÁMETROS:")
            for error in errores:
                print(f"   - {error}")
            return False
        else:
            return True

    def inicializar_individuo(self):
        individuo = []
        todas_posiciones = [(f, c) for f in range(8) for c in range(8)]
        np.random.shuffle(todas_posiciones)
        for i in range(self.num_piezas):
            fila, columna = todas_posiciones[i]
            tipo_index = np.random.randint(0, len(self.piezas_disponibles))
            tipo_bits = [int(x) for x in format(tipo_index, "02b")]
            fila_bits = [int(x) for x in format(fila, "03b")]
            col_bits = [int(x) for x in format(columna, "03b")]
            individuo += tipo_bits + fila_bits + col_bits
        return individuo

    def inicializar_poblacion(self, tam_poblacion):
        poblacion=[]
        for i in range(tam_poblacion):
            individuo=self.inicializar_individuo()
            poblacion.append(individuo)
        return poblacion

    def reparar_superposiciones(self, individuo):
        posiciones_ocupadas = set()
        individuo_reparado = individuo.copy()

        for i in range(self.num_piezas):
            tipo_bits = individuo_reparado[i * 8: i * 8 + 2]
            fila_bits = individuo_reparado[i * 8 + 2: i * 8 + 5]
            col_bits = individuo_reparado[i * 8 + 5: i * 8 + 8]

            fila = int(''.join(str(b) for b in fila_bits), 2)
            columna = int(''.join(str(b) for b in col_bits), 2)
            posicion = (fila, columna)

            if posicion in posiciones_ocupadas:
                todas_posiciones = [(f, c) for f in range(8) for c in range(8)]
                np.random.shuffle(todas_posiciones)
                for nueva_fila, nueva_columna in todas_posiciones:
                    if (nueva_fila, nueva_columna) not in posiciones_ocupadas:
                        nueva_fila_bits = [int(x) for x in format(nueva_fila, "03b")]
                        nueva_col_bits = [int(x) for x in format(nueva_columna, "03b")]
                        individuo_reparado[i * 8 + 2: i * 8 + 5] = nueva_fila_bits
                        individuo_reparado[i * 8 + 5: i * 8 + 8] = nueva_col_bits
                        posicion = (nueva_fila, nueva_columna)
                        break

            posiciones_ocupadas.add(posicion)

        return individuo_reparado
    def evaluar_poblacion(self, individuo):
        tablero_temp = chess.Board.empty()
        for i in range(self.num_piezas):
            tipo_bits = individuo[i * 8: i * 8 + 2]
            fila_bits = individuo[i * 8 + 2: i * 8 + 5]
            col_bits = individuo[i * 8 + 5: i * 8 + 8]

            tipo_index = int(''.join(str(b) for b in tipo_bits), 2)
            fila = int(''.join(str(b) for b in fila_bits), 2)
            columna = int(''.join(str(b) for b in col_bits), 2)

            pieza = self.piezas_disponibles[tipo_index]

            tablero_temp.set_piece_at(chess.square(columna, fila), chess.Piece(pieza, chess.WHITE))#Coloca las piezas en el tablero

        cobertura = set()
        for square, piece in tablero_temp.piece_map().items():
            cobertura.update(tablero_temp.attacks(square))
        fitness = self.fitness_valor_por_cobertura(tablero_temp)
        return fitness, len(cobertura)
    def seleccion_torneo(self, poblacion, fitness, tam_torneo=3):
        seleccionados = []
        for i in range(len(poblacion)):#Selecciona un padre por individuo
            indices = np.random.choice(len(poblacion), tam_torneo, replace=False)
            competidores = fitness[indices]
            ganador_idx = indices[np.argmax(competidores)]
            seleccionados.append(poblacion[ganador_idx])
        return np.array(seleccionados)
    def seleccion_ruleta(self, poblacion, fitness):
        adjusted_fitness = np.max(fitness) - fitness + 1
        prob = adjusted_fitness / np.sum(adjusted_fitness)
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

                hijo1 = self.reparar_superposiciones(hijo1)
                hijo2 = self.reparar_superposiciones(hijo2)
            else:
                hijo1, hijo2 = padre1.copy(), padre2.copy()

            hijos.append(hijo1)
            hijos.append(hijo2)
        return np.array(hijos[:self.tam_poblacion])

    def mutacion(self, poblacion):
        for i in range(len(poblacion)):
            individuo_mutado = poblacion[i].copy()
            for j in range(len(individuo_mutado)):
                if random.random() < self.prob_mutacion:
                    individuo_mutado[j] = 1 - individuo_mutado[j]

            poblacion[i] = self.reparar_superposiciones(individuo_mutado)
        return poblacion
    def colocar_individuo_en_tablero(self,individuo):
        self.logic_board = chess.Board.empty()
        for i in range(self.num_piezas):
            tipo_bits = individuo[i * 8: i * 8 + 2]
            fila_bits = individuo[i * 8 + 2: i * 8 + 5]
            col_bits = individuo[i * 8 + 5: i * 8 + 8]

            tipo_index = int("".join(str(b) for b in tipo_bits), 2)
            fila = int("".join(str(b) for b in fila_bits), 2)
            columna = int("".join(str(b) for b in col_bits), 2)

            pieza = self.piezas_disponibles[tipo_index]
            self.poner_pieza(pieza, fila, columna)
    def fitness_valor_por_cobertura(self, tablero_temp):
        conteo_piezas = {pieza: 0 for pieza in self.piezas_disponibles}
        valor_total = 0
        for square, piece in tablero_temp.piece_map().items():
            conteo_piezas[piece.piece_type] += 1
            valor_total += self.valor_piezas[piece.piece_type]

        penalizacion = 0
        for pieza, limite in self.limites_piezas.items():
            if conteo_piezas[pieza] > limite:
                penalizacion += (conteo_piezas[pieza] - limite) * 50

        cobertura_total = set()
        for square, piece in tablero_temp.piece_map().items():
            ataques = tablero_temp.attacks(square)
            cobertura_total.update(ataques)

        return self.peso_cobertura*len(cobertura_total) - self.peso_valor_piezas*valor_total - self.peso_penalizacion*penalizacion

    def loop(self):
        running = True
        clock = pygame.time.Clock()

        poblacion = self.inicializar_poblacion(self.tam_poblacion)

        mejor_global = None
        mejor_fitness = -9999999
        mejor_iteracion = 0
        lista_mejor = []

        for iteracion in range(self.max_iter):
            resultados = [self.evaluar_poblacion(ind) for ind in poblacion]
            fitness_poblacion = np.array([resultado[0] for resultado in resultados])
            cobertura_poblacion = [resultado[1] for resultado in resultados]
            best_idx = np.argmax(fitness_poblacion)

            if fitness_poblacion[best_idx] > mejor_fitness:
                mejor_fitness = fitness_poblacion[best_idx]
                mejor_global = poblacion[best_idx].copy()
                mejor_iteracion = iteracion

                # ACTUALIZAR ESTADÍSTICAS PARA EL PANEL
                self.mejor_fitness_global = mejor_fitness
                self.mejor_iteracion_global = mejor_iteracion
                self.casillas_controladas_global = cobertura_poblacion[best_idx]

            print(
                f"Iteración {iteracion:2d}: Fitness = {fitness_poblacion[best_idx]:6.2f}, Casillas controladas = {cobertura_poblacion[best_idx]}/64")

            padres = self.seleccion_padres(np.array(poblacion, dtype=object), fitness_poblacion)
            hijos = self.cruzamiento(padres.tolist())
            poblacion = self.mutacion(hijos)
            lista_mejor.append(np.max(fitness_poblacion))

            clock.tick(60)

        resultados_final = self.evaluar_poblacion(mejor_global)
        self.casillas_controladas_global = resultados_final[1]

        print(
            f"Mejor solución en iteración: {mejor_iteracion}, Mejor fitness: {mejor_fitness}, Casillas controladas: {self.casillas_controladas_global}/64")

        self.colocar_individuo_en_tablero(mejor_global)

        # plt.plot(lista_mejor, label="Mejor fitness")
        # plt.legend()
        # plt.xlabel("Generación")
        # plt.ylabel("Fitness")
        # plt.title("Evolución del algoritmo genético")
        # plt.grid()
        # plt.show()

        self.render()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

max_iter = 50
tam_poblacion = 20
prob_cruza = 0.95
prob_mutacion = 0.9
num_piezas=10
porc_ruleta=0.5
porc_torneo=0.5
peso_cobertura=0.5
peso_valor_piezas=0.3
peso_penalizacion=0.2


juego = Chess(
    max_iter=max_iter,
    tam_poblacion=tam_poblacion,
    prob_cruza=prob_cruza,
    prob_mutacion=prob_mutacion,
    num_piezas=num_piezas,
    porc_ruleta=porc_ruleta,
    porc_torneo=porc_torneo,
    peso_cobertura=peso_cobertura,
    peso_valor_piezas=peso_valor_piezas,
    peso_penalizacion=peso_penalizacion
)

pygame.quit()