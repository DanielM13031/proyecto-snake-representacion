"""
Módulo agentes.py
Implementa la jerarquía de agentes siguiendo la interfaz de los talleres del curso
(agentes.py del Taller 10 y agents.py del Taller 9):
1. Agent: clase base del curso (make_decision, program, restart).
2. Random: agente aleatorio de línea base.
3. AgenteSnake: agente basado en objetivos con soporte para los 6 algoritmos de
   búsqueda, heurísticas, costo de riesgo y chequeo de seguridad (camino a la cola).
4. AgenteFrutaAdversaria: agente Minimax con poda Alfa-Beta para la variante
   de fruta adversaria (Sección 5).
"""

import math
import random
import time
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

from ambientes import (
    Celda,
    EstadoSnake,
    SubproblemaSnake,
    SnakeJuegoAdversario,
    obtener_acciones_aplicables,
    aplicar_transicion,
    calcular_riesgo_celda,
)
from busqueda import (
    resolver_subproblema,
    anchura,
    heuristica_manhattan,
    CATALOGO_HEURISTICAS,
)


# =============================================================================
# 1. CLASE BASE Agent Y Random (ESTÁNDAR DE LOS TALLERES DEL PROFESOR)
# =============================================================================

class Agent:
    """
    Defines the basic methods for the agent (idéntica a agentes.py de los talleres).
    """

    def __init__(self):
        self.plan: List[Any] = []
        self.states: List[Any] = []
        self.actions: List[Any] = []
        self.rewards: List[float] = [np.nan]
        self.dones: List[bool] = [False]
        self.turn: int = 0

    def make_decision(self) -> Any:
        """
        Agent makes a decision according to its model.
        """
        if len(self.plan) == 0:
            self.program()
        try:
            action = self.plan.pop(0)
        except Exception:
            state = self.states[-1] if self.states else None
            raise Exception(f"¡Plan vacío! Revisar reglas en estado {state}")
        self.turn += 1
        return action

    def program(self) -> None:
        """
        Debe ser modificada por cada subclase.
        """
        pass

    def reset(self) -> None:
        self.restart()

    def restart(self) -> None:
        """
        Restarts the agent for a new trial.
        """
        self.plan = []
        self.states = []
        self.actions = []
        self.rewards = [np.nan]
        self.dones = [False]
        self.turn = 0


class Random(Agent):
    """Agente que selecciona acciones aplicables al azar."""

    def __init__(self, choices: Optional[List[str]] = None):
        super().__init__()
        self.choices = choices if choices is not None else ['norte', 'sur', 'este', 'oeste']

    def program(self) -> None:
        if self.states and isinstance(self.states[-1], EstadoSnake):
            aplicables = obtener_acciones_aplicables(self.states[-1])
            if aplicables:
                self.plan.append(random.choice(aplicables))
                return
        self.plan.append(random.choice(self.choices))


# =============================================================================
# 2. AGENTE BASADO EN OBJETIVOS PARA SNAKE (SECCIÓN 3 DE LA PROPUESTA)
# =============================================================================

class AgenteSnake(Agent):
    """
    Agente basado en objetivos que alterna planeación y ejecución por cada fruta.
    - Resuelve cada subproblema con búsqueda en grafo (BFS, DFS, UCS, Avara, A*, Beam).
    - Si chequeo_seguridad=True, antes de comprometerse con el camino hacia la fruta,
      simula el estado final al comer y verifica que exista un camino desde la nueva
      cabeza hasta la cola c_L. Si no es seguro, da un paso persiguiendo su cola
      por la ruta que mejor preserve el espacio libre y vuelve a planear.
    """

    def __init__(
        self,
        algoritmo: str = "A*",
        heuristica: str = "manhattan",
        chequeo_seguridad: bool = True,
        lambda_riesgo: float = 0.0,
        k_haz: int = 3,
        replanear_cada_paso: bool = False,
    ):
        super().__init__()
        self.algoritmo = algoritmo
        self.heuristica = heuristica
        self.chequeo_seguridad = chequeo_seguridad
        self.lambda_riesgo = lambda_riesgo
        self.k_haz = k_haz
        self.replanear_cada_paso = replanear_cada_paso

        # Telemetría acumulada por partida (para evaluación experimental)
        self.historial_decisiones: List[Dict[str, Any]] = []
        self.veces_camino_inseguro: int = 0
        self.pasos_siguiendo_cola: int = 0
        self._fruta_objetivo_actual: Optional[Tuple[Celda, ...]] = None

    def restart(self) -> None:
        super().restart()
        self.historial_decisiones = []
        self.veces_camino_inseguro = 0
        self.pasos_siguiendo_cola = 0
        self._fruta_objetivo_actual = None

    def make_decision(self) -> str:
        """
        Si cambió la fruta en el tablero o si replanear_cada_paso=True,
        limpia el plan actual para forzar una nueva planeación en program().
        """
        if self.states:
            estado_actual: EstadoSnake = self.states[-1]
            frutas_tablero = estado_actual.todas_las_frutas
            if self.replanear_cada_paso or (frutas_tablero != self._fruta_objetivo_actual):
                self.plan = []
                self._fruta_objetivo_actual = frutas_tablero

            # Verificación defensiva: si la siguiente acción del plan dejó de ser válida
            if self.plan:
                acciones_validas = obtener_acciones_aplicables(estado_actual)
                if self.plan[0] not in acciones_validas:
                    self.plan = []

        return super().make_decision()

    @staticmethod
    def _tiene_salida_a_cola(estado: EstadoSnake) -> Tuple[bool, int, int, int, int]:
        """
        Verifica mediante BFS en grafo si desde 'estado' existe un camino válido
        desde la cabeza c_1 hasta la cola c_L (sin comer fruta en el destino).
        Retorna: (es_seguro, distancia_a_cola, nodos_gen, nodos_exp, max_front).
        """
        if estado.longitud <= 2:
            acciones = obtener_acciones_aplicables(estado)
            return (len(acciones) > 0, 1, 1, 0, 1)

        prob_cola = SubproblemaSnake(
            estado_inicial=estado,
            objetivo=estado.cola,
            lambda_riesgo=0.0,
            comer_en_objetivo=False,
        )
        anchura(prob_cola, guardar_traza=False)
        res = prob_cola.ultimo_resultado
        if res.exito:
            return (
                True,
                res.longitud_solucion,
                res.nodos_generados,
                res.nodos_expandidos,
                res.max_frontera,
            )
        return (False, 0, res.nodos_generados, res.nodos_expandidos, res.max_frontera)

    def _elegir_paso_hacia_cola(
        self, estado: EstadoSnake
    ) -> Tuple[str, int, int, int]:
        """
        Maniobra de emergencia cuando el camino a la fruta no existe o no es seguro.
        Evalúa las acciones aplicables desde 'estado', filtra aquellas desde las cuales
        la serpiente mantiene un camino válido hacia su propia cola y elige la que
        MAXIMIZA la distancia hacia la cola (y minimiza el riesgo local) para hacer
        tiempo y desenrollar el cuerpo sin encerrarse.
        """
        acciones_validas = obtener_acciones_aplicables(estado)
        if not acciones_validas:
            # Sin acciones aplicables: devuelve cualquier dirección para que el entorno cierre
            return ('norte', 0, 0, 0)

        total_gen = 0
        total_exp = 0
        max_front = 0

        candidatas_seguras: List[Tuple[float, str]] = []
        candidatas_respaldo: List[Tuple[float, str]] = []

        for acc in acciones_validas:
            # Simulamos dar 1 paso con esta acción
            est_sig = aplicar_transicion(estado, acc, comer_fruta=True)
            seguro, dist_cola, ng, ne, mf = self._tiene_salida_a_cola(est_sig)
            total_gen += ng
            total_exp += ne
            max_front = max(max_front, mf)

            riesgo = calcular_riesgo_celda(est_sig)
            dist_fruta = (
                heuristica_manhattan(est_sig, estado.fruta)
                if estado.fruta is not None
                else 0.0
            )
            # Puntaje para seguimiento de cola: preferimos mayor distancia a la cola
            # (para no pegarnos a ella de inmediato), menor riesgo local y acercarnos a la fruta
            puntaje = (dist_cola * 10.0) - (riesgo * 3.0) - (dist_fruta * 0.2)

            if seguro:
                candidatas_seguras.append((puntaje, acc))
            else:
                # Si ninguna mantiene salida a la cola, preferimos la de menor riesgo local
                candidatas_respaldo.append((-riesgo, acc))

        if candidatas_seguras:
            candidatas_seguras.sort(key=lambda x: x[0], reverse=True)
            return (candidatas_seguras[0][1], total_gen, total_exp, max_front)

        candidatas_respaldo.sort(key=lambda x: x[0], reverse=True)
        return (candidatas_respaldo[0][1], total_gen, total_exp, max_front)

    def program(self) -> None:
        """
        Genera el plan de acción para el estado actual s = (C, f).
        """
        t_inicio = time.perf_counter()
        estado: EstadoSnake = self.states[-1]
        self._fruta_objetivo_actual = estado.todas_las_frutas

        # 1. Búsqueda principal hacia la fruta
        prob_fruta = SubproblemaSnake(
            estado_inicial=estado,
            objetivo=estado.fruta,
            lambda_riesgo=self.lambda_riesgo,
            comer_en_objetivo=True,
        )
        res_fruta = resolver_subproblema(
            prob_fruta,
            algoritmo=self.algoritmo,
            heuristica=self.heuristica,
            k_haz=self.k_haz,
            guardar_traza=False,
        )

        nodos_gen = res_fruta.nodos_generados
        nodos_exp = res_fruta.nodos_expandidos
        max_front = res_fruta.max_frontera
        modo_decision = "ruta_fruta"

        if res_fruta.exito and len(res_fruta.acciones) > 0:
            if not self.chequeo_seguridad:
                # Sin chequeo de seguridad: ejecuta todo el camino planeado hacia la fruta
                self.plan = list(res_fruta.acciones)
            else:
                # Con chequeo de seguridad: simulamos el estado al terminar de comer la fruta
                estado_post_fruta = res_fruta.estado_final
                es_seguro, _, ng_c, ne_c, mf_c = self._tiene_salida_a_cola(
                    estado_post_fruta
                )
                nodos_gen += ng_c
                nodos_exp += ne_c
                max_front = max(max_front, mf_c)

                if es_seguro:
                    self.plan = list(res_fruta.acciones)
                else:
                    # El camino directo a la fruta encierra a la serpiente -> seguir la cola 1 paso
                    self.veces_camino_inseguro += 1
                    self.pasos_siguiendo_cola += 1
                    modo_decision = "evasion_cola (ruta insegura)"
                    acc_cola, ng_e, ne_e, mf_e = self._elegir_paso_hacia_cola(estado)
                    nodos_gen += ng_e
                    nodos_exp += ne_e
                    max_front = max(max_front, mf_e)
                    self.plan = [acc_cola]
        else:
            # No se encontró camino hacia la fruta -> avanzar 1 paso siguiendo la cola
            self.pasos_siguiendo_cola += 1
            modo_decision = "evasion_cola (sin camino a fruta)"
            acc_cola, ng_e, ne_e, mf_e = self._elegir_paso_hacia_cola(estado)
            nodos_gen += ng_e
            nodos_exp += ne_e
            max_front = max(max_front, mf_e)
            self.plan = [acc_cola]

        tiempo_total_ms = (time.perf_counter() - t_inicio) * 1000.0
        self.historial_decisiones.append(
            {
                "turno": self.turn,
                "longitud_serpiente": estado.longitud,
                "modo": modo_decision,
                "exito_fruta": res_fruta.exito,
                "nodos_generados": nodos_gen,
                "nodos_expandidos": nodos_exp,
                "max_frontera": max_front,
                "costo_plan": res_fruta.costo_total if res_fruta.exito else np.nan,
                "longitud_plan": res_fruta.longitud_solucion if res_fruta.exito else 1,
                "tiempo_ms": tiempo_total_ms,
            }
        )


# =============================================================================
# 3. AGENTE PARA VARIANTE DE FRUTA ADVERSARIA (MINIMAX CON PODA ALFA-BETA)
# =============================================================================

class AgenteFrutaAdversaria(Agent):
    """
    Implementa la variante de Fruta Adversaria de la Sección 5 usando Minimax
    con poda Alfa-Beta y función de evaluación heurística (al estilo de
    ChessFinalist1 en agents.py del Taller 9).
    Puede actuar como el oponente 'fruta' (MIN, que coloca la manzana en la peor
    posición posible para la serpiente) o como 'serpiente' (MAX).
    """

    def __init__(
        self,
        game: SnakeJuegoAdversario,
        jugador: str = 'fruta',
        max_lim: int = 3,
        max_candidatas_fruta: int = 12,
    ):
        super().__init__()
        self.game = game
        self.jugador = jugador
        self.max_lim = max_lim
        self.max_candidatas_fruta = max_candidatas_fruta
        self.nodos_evaluados_minimax: int = 0
        self.podas_alfa_beta: int = 0

    def program(self) -> None:
        estado: EstadoSnake = self.states[-1]
        _, accion = self._H_minimax_alfa_beta(estado, 0, -math.inf, math.inf)
        if accion is None:
            acciones = self.game.acciones(estado)
            accion = acciones[0] if acciones else 'norte'
        self.plan.append(accion)

    def _acciones_acotadas(self, estado: EstadoSnake) -> List[Any]:
        """
        Para que la rama del jugador 'fruta' no evalúe 90 casillas irrelevantes,
        prioriza las celdas libres más alejadas de la cabeza o más rodeadas por muros/cuerpo.
        """
        acciones = self.game.acciones(estado)
        if estado.turno == 'serpiente' or len(acciones) <= self.max_candidatas_fruta:
            return acciones

        xh, yh = estado.cabeza
        ocupadas = set(estado.cuerpo) | set(estado.obstaculos)

        def dificultad_celda(celda: Celda) -> float:
            cx, cy = celda
            dist = abs(xh - cx) + abs(yh - cy)
            vecinos_bloq = 0
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < estado.tamano and 0 <= ny < estado.tamano) or (nx, ny) in ocupadas:
                    vecinos_bloq += 1
            # A la fruta adversaria le interesa mucha distancia y muchos vecinos bloqueados
            return dist + 3.0 * vecinos_bloq

        acciones_ordenadas = sorted(acciones, key=dificultad_celda, reverse=True)
        return acciones_ordenadas[: self.max_candidatas_fruta]

    def _is_cutoff(self, estado: EstadoSnake, d: int) -> bool:
        if self.game.es_terminal(estado):
            return True
        return d >= self.max_lim

    def _evaluar(self, estado: EstadoSnake) -> float:
        """
        Función de utilidad/evaluación desde la perspectiva de la serpiente (MAX):
        - Ganancia por longitud alcanzada y por grados de libertad (acciones disponibles).
        - Pérdida por distancia a la fruta y por riesgo de encierro de la fruta.
        """
        self.nodos_evaluados_minimax += 1
        if self.game.es_terminal(estado):
            util = self.game.utilidad(estado, 'serpiente')
            return float(util) if util is not None else 0.0

        libertad_cabeza = len(obtener_acciones_aplicables(estado))
        riesgo_cabeza = calcular_riesgo_celda(estado)

        dist_fruta = 0.0
        encierro_fruta = 0.0
        if estado.fruta is not None:
            fx, fy = estado.fruta
            xh, yh = estado.cabeza
            dist_fruta = float(abs(xh - fx) + abs(yh - fy))
            ocupadas = set(estado.cuerpo) | set(estado.obstaculos)
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nx, ny = fx + dx, fy + dy
                if not (0 <= nx < estado.tamano and 0 <= ny < estado.tamano) or (nx, ny) in ocupadas:
                    encierro_fruta += 1.0

        # Utilidad para MAX (serpiente): quiere alta longitud, alta libertad,
        # baja distancia a la fruta y fruta en lugares despejados.
        utilidad_max = (
            (50.0 * estado.longitud)
            + (8.0 * libertad_cabeza)
            - (10.0 * riesgo_cabeza)
            - (2.5 * dist_fruta)
            - (6.0 * encierro_fruta)
        )
        return utilidad_max

    def _H_minimax_alfa_beta(
        self, estado: EstadoSnake, d: int, alfa: float, beta: float
    ) -> Tuple[float, Any]:
        """
        Búsqueda H-Minimax con poda Alfa-Beta (misma estructura de agents.py del Taller 9).
        """
        jugador_turno = self.game.player(estado)
        if self._is_cutoff(estado, d):
            return self._evaluar(estado), None

        acciones = self._acciones_acotadas(estado)
        if not acciones:
            return self._evaluar(estado), None

        if jugador_turno == 'serpiente':  # Jugador MAX
            v = -math.inf
            mejor_accion = acciones[0]
            for a in acciones:
                est_res = self.game.resultado(estado, a)
                v2, _ = self._H_minimax_alfa_beta(est_res, d + 1, alfa, beta)
                if v2 > v:
                    v = v2
                    mejor_accion = a
                    alfa = max(alfa, v)
                if v >= beta:
                    self.podas_alfa_beta += 1
                    return v, mejor_accion
            return v, mejor_accion

        elif jugador_turno == 'fruta':  # Jugador MIN (Fruta Adversaria)
            v = math.inf
            mejor_accion = acciones[0]
            for a in acciones:
                est_res = self.game.resultado(estado, a)
                v2, _ = self._H_minimax_alfa_beta(est_res, d + 1, alfa, beta)
                if v2 < v:
                    v = v2
                    mejor_accion = a
                    beta = min(beta, v)
                if v <= alfa:
                    self.podas_alfa_beta += 1
                    return v, mejor_accion
            return v, mejor_accion

        else:
            raise ValueError(f"Jugador desconocido: {jugador_turno}")