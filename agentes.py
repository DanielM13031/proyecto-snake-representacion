"""Agentes que juegan Snake.

El módulo define la jerarquía de agentes del proyecto:

1. ``Agent``: clase base con el ciclo ``make_decision``, ``program`` y
   ``restart``.
2. ``Random``: agente aleatorio que sirve como línea base.
3. ``AgenteSnake``: agente basado en modelo y en objetivos que resuelve
   cada fruta como un subproblema de búsqueda y aplica el chequeo de
   seguridad hacia la cola.
4. ``AgenteFrutaAdversaria``: agente Minimax con poda alfa-beta para la
   variante en la que la fruta la coloca un oponente.
"""

import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ambientes import (
    Celda,
    EstadoSnake,
    SnakeJuegoAdversario,
    SubproblemaSnake,
    aplicar_transicion,
    calcular_riesgo_celda,
    obtener_acciones_aplicables,
)
from busqueda import (
    CATALOGO_HEURISTICAS,
    anchura,
    heuristica_manhattan,
    resolver_subproblema,
)

# =============================================================================
# 1. Clase base y agente aleatorio
# =============================================================================


class Agent:
    """Clase base de los agentes.

    Guarda el historial de estados, acciones y recompensas de una
    partida y un plan de acciones pendientes. Las subclases solo deben
    implementar ``program``, que llena el plan cuando se vacía.
    """

    def __init__(self):
        """Inicializa un agente sin plan ni historial."""
        self.plan: List[Any] = []
        self.states: List[Any] = []
        self.actions: List[Any] = []
        self.rewards: List[float] = [np.nan]
        self.dones: List[bool] = [False]
        self.turn: int = 0

    def make_decision(self) -> Any:
        """Devuelve la siguiente acción del plan.

        Si el plan está vacío, llama a ``program`` para generar uno
        nuevo.

        Raises
        ------
        Exception
            Si ``program`` no deja ninguna acción en el plan.
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
        """Genera un plan para el estado actual.

        Cada subclase debe sobrescribir este método.
        """
        pass

    def reset(self) -> None:
        """Reinicia el agente; es un alias de ``restart``."""
        self.restart()

    def restart(self) -> None:
        """Borra el plan y el historial para una partida nueva."""
        self.plan = []
        self.states = []
        self.actions = []
        self.rewards = [np.nan]
        self.dones = [False]
        self.turn = 0


class Random(Agent):
    """Agente que elige al azar entre las acciones aplicables."""

    def __init__(self, choices: Optional[List[str]] = None):
        """Crea el agente.

        Parameters
        ----------
        choices : list of str, optional
            Acciones entre las que elegir cuando el estado no es un
            ``EstadoSnake``. Por defecto, las cuatro direcciones.
        """
        super().__init__()
        self.choices = (
            choices
            if choices is not None
            else ["norte", "sur", "este", "oeste"]
        )

    def program(self) -> None:
        """Agrega al plan una acción aplicable elegida al azar."""
        if self.states and isinstance(self.states[-1], EstadoSnake):
            aplicables = obtener_acciones_aplicables(self.states[-1])
            if aplicables:
                self.plan.append(random.choice(aplicables))
                return
        self.plan.append(random.choice(self.choices))


# =============================================================================
# 2. Agente basado en objetivos para Snake
# =============================================================================


class AgenteSnake(Agent):
    """Agente basado en modelo y en objetivos para Snake.

    Por cada fruta resuelve un subproblema de búsqueda en grafo con el
    algoritmo elegido (BFS, DFS, UCS, Avara, A* o Beam) y ejecuta el
    plan encontrado. Con ``chequeo_seguridad=True``, antes de aceptar un
    plan simula el estado en el que quedaría al comer y verifica que la
    cabeza todavía pueda llegar a la cola. Si no puede, da un solo paso
    siguiendo su cola y vuelve a planear.

    Parameters
    ----------
    algoritmo : str
        Nombre del algoritmo de búsqueda, por ejemplo ``"A*"``.
    heuristica : str
        Clave de la heurística en ``CATALOGO_HEURISTICAS``.
    chequeo_seguridad : bool
        Si se verifica el camino hacia la cola antes de ejecutar un
        plan.
    lambda_riesgo : float
        Peso del riesgo local en el costo por paso.
    k_haz : int
        Ancho del haz para la búsqueda en haz.
    replanear_cada_paso : bool
        Si se descarta el plan y se vuelve a planear en cada turno.
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
        """Crea el agente con la configuración indicada."""
        super().__init__()
        self.algoritmo = algoritmo
        self.heuristica = heuristica
        self.chequeo_seguridad = chequeo_seguridad
        self.lambda_riesgo = lambda_riesgo
        self.k_haz = k_haz
        self.replanear_cada_paso = replanear_cada_paso

        # Métricas acumuladas durante la partida para la evaluación.
        self.historial_decisiones: List[Dict[str, Any]] = []
        self.veces_camino_inseguro: int = 0
        self.pasos_siguiendo_cola: int = 0
        self._fruta_objetivo_actual: Optional[Tuple[Celda, ...]] = None

    def restart(self) -> None:
        """Reinicia el agente y sus métricas para una partida nueva."""
        super().restart()
        self.historial_decisiones = []
        self.veces_camino_inseguro = 0
        self.pasos_siguiendo_cola = 0
        self._fruta_objetivo_actual = None

    def make_decision(self) -> str:
        """Devuelve la siguiente acción, replaneando cuando hace falta.

        El plan se descarta si cambiaron las frutas del tablero, si
        ``replanear_cada_paso`` está activo o si la siguiente acción del
        plan dejó de ser aplicable.
        """
        if self.states:
            estado_actual: EstadoSnake = self.states[-1]
            frutas_tablero = estado_actual.todas_las_frutas
            if self.replanear_cada_paso or (
                frutas_tablero != self._fruta_objetivo_actual
            ):
                self.plan = []
                self._fruta_objetivo_actual = frutas_tablero

            # Verificación defensiva: el plan se descarta si su
            # siguiente acción ya no es aplicable en el estado actual.
            if self.plan:
                acciones_validas = obtener_acciones_aplicables(estado_actual)
                if self.plan[0] not in acciones_validas:
                    self.plan = []

        return super().make_decision()

    @staticmethod
    def _tiene_salida_a_cola(
        estado: EstadoSnake,
    ) -> Tuple[bool, int, int, int, int]:
        """Verifica con BFS si la cabeza puede llegar a la cola.

        Parameters
        ----------
        estado : EstadoSnake
            Estado en el que se hace la verificación.

        Returns
        -------
        tuple
            ``(es_seguro, distancia_a_cola, nodos_generados,
            nodos_expandidos, frontera_maxima)``.
        """
        # Con longitud 1 o 2 la cola está pegada a la cabeza: basta con
        # tener alguna acción aplicable.
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
        return (
            False,
            0,
            res.nodos_generados,
            res.nodos_expandidos,
            res.max_frontera,
        )

    def _elegir_paso_hacia_cola(
        self, estado: EstadoSnake
    ) -> Tuple[str, int, int, int]:
        """Elige un paso de evasión cuando ir a la fruta no es seguro.

        Prueba cada acción aplicable, se queda con las que mantienen un
        camino hasta la cola y entre ellas prefiere la que deja la cola
        más lejos, tiene menor riesgo local y se acerca a la fruta. Así
        la serpiente gana tiempo y desenrolla el cuerpo sin encerrarse.

        Parameters
        ----------
        estado : EstadoSnake
            Estado actual de la serpiente.

        Returns
        -------
        tuple
            ``(accion, nodos_generados, nodos_expandidos,
            frontera_maxima)``.
        """
        acciones_validas = obtener_acciones_aplicables(estado)
        if not acciones_validas:
            # Sin acciones aplicables se devuelve cualquier dirección
            # para que el entorno registre el fin de la partida.
            return ("norte", 0, 0, 0)

        total_gen = 0
        total_exp = 0
        max_front = 0

        candidatas_seguras: List[Tuple[float, str]] = []
        candidatas_respaldo: List[Tuple[float, str]] = []

        for acc in acciones_validas:
            # Se simula un paso con la acción y se revisa la salida a
            # cola.
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
            # Se prefiere mayor distancia a la cola, para no pegarse a
            # ella, menor riesgo local y menor distancia a la fruta.
            puntaje = (dist_cola * 10.0) - (riesgo * 3.0) - (dist_fruta * 0.2)

            if seguro:
                candidatas_seguras.append((puntaje, acc))
            else:
                # Respaldo por si ninguna acción conserva la salida a la
                # cola: se ordena solo por menor riesgo local.
                candidatas_respaldo.append((-riesgo, acc))

        if candidatas_seguras:
            candidatas_seguras.sort(key=lambda x: x[0], reverse=True)
            return (candidatas_seguras[0][1], total_gen, total_exp, max_front)

        candidatas_respaldo.sort(key=lambda x: x[0], reverse=True)
        return (candidatas_respaldo[0][1], total_gen, total_exp, max_front)

    def program(self) -> None:
        """Genera el plan de acciones para el estado actual s = (C, f).

        Busca un camino hasta la fruta. Si existe y, cuando corresponde,
        pasa el chequeo de seguridad, el plan es el camino completo. En
        otro caso el plan es un único paso de evasión. Cada decisión
        queda registrada en ``historial_decisiones``.
        """
        t_inicio = time.perf_counter()
        estado: EstadoSnake = self.states[-1]
        self._fruta_objetivo_actual = estado.todas_las_frutas

        # 1. Búsqueda principal hacia la fruta.
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
                # 2a. Sin chequeo se ejecuta todo el camino a la fruta.
                self.plan = list(res_fruta.acciones)
            else:
                # 2b. Con chequeo se simula el estado al comer la fruta
                # y se verifica que siga habiendo camino hasta la cola.
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
                    # El camino directo encerraría a la serpiente: se da
                    # un solo paso siguiendo la cola.
                    self.veces_camino_inseguro += 1
                    self.pasos_siguiendo_cola += 1
                    modo_decision = "evasion_cola (ruta insegura)"
                    acc_cola, ng_e, ne_e, mf_e = self._elegir_paso_hacia_cola(
                        estado
                    )
                    nodos_gen += ng_e
                    nodos_exp += ne_e
                    max_front = max(max_front, mf_e)
                    self.plan = [acc_cola]
        else:
            # 3. No hay camino a la fruta: un paso siguiendo la cola.
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
                "costo_plan": (
                    res_fruta.costo_total if res_fruta.exito else np.nan
                ),
                "longitud_plan": (
                    res_fruta.longitud_solucion if res_fruta.exito else 1
                ),
                "tiempo_ms": tiempo_total_ms,
            }
        )


# =============================================================================
# 3. Agente para la variante de fruta adversaria
# =============================================================================


class AgenteFrutaAdversaria(Agent):
    """Agente Minimax con poda alfa-beta para la fruta adversaria.

    Puede jugar como ``"fruta"``, el jugador MIN que coloca cada fruta
    en la peor celda posible para la serpiente, o como ``"serpiente"``,
    el jugador MAX. La búsqueda se corta a profundidad ``max_lim`` y los
    estados no terminales se valoran con una función de evaluación.

    Parameters
    ----------
    game : SnakeJuegoAdversario
        Juego bipersonal sobre el que se busca.
    jugador : str
        ``"fruta"`` o ``"serpiente"``.
    max_lim : int
        Profundidad máxima de la búsqueda.
    max_candidatas_fruta : int
        Número máximo de celdas que se evalúan en los turnos de la
        fruta.
    """

    def __init__(
        self,
        game: SnakeJuegoAdversario,
        jugador: str = "fruta",
        max_lim: int = 3,
        max_candidatas_fruta: int = 12,
    ):
        """Crea el agente adversario con la configuración indicada."""
        super().__init__()
        self.game = game
        self.jugador = jugador
        self.max_lim = max_lim
        self.max_candidatas_fruta = max_candidatas_fruta
        self.nodos_evaluados_minimax: int = 0
        self.podas_alfa_beta: int = 0

    def program(self) -> None:
        """Agrega al plan la mejor acción según Minimax alfa-beta."""
        estado: EstadoSnake = self.states[-1]
        _, accion = self._H_minimax_alfa_beta(estado, 0, -math.inf, math.inf)
        if accion is None:
            acciones = self.game.acciones(estado)
            accion = acciones[0] if acciones else "norte"
        self.plan.append(accion)

    def _acciones_acotadas(self, estado: EstadoSnake) -> List[Any]:
        """Devuelve las acciones a explorar, acotadas para la fruta.

        Para no evaluar decenas de celdas irrelevantes, en el turno de
        la fruta se conservan solo las ``max_candidatas_fruta`` celdas
        más difíciles para la serpiente: las más lejanas de la cabeza y
        las más rodeadas por muros o por el cuerpo.
        """
        acciones = self.game.acciones(estado)
        if (
            estado.turno == "serpiente"
            or len(acciones) <= self.max_candidatas_fruta
        ):
            return acciones

        xh, yh = estado.cabeza
        ocupadas = set(estado.cuerpo) | set(estado.obstaculos)

        def dificultad_celda(celda: Celda) -> float:
            """Mide cuánto le cuesta a la serpiente llegar a celda."""
            cx, cy = celda
            dist = abs(xh - cx) + abs(yh - cy)
            vecinos_bloq = 0
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nx, ny = cx + dx, cy + dy
                if (
                    not (0 <= nx < estado.tamano and 0 <= ny < estado.tamano)
                    or (nx, ny) in ocupadas
                ):
                    vecinos_bloq += 1
            # La fruta adversaria prefiere celdas lejanas y encerradas.
            return dist + 3.0 * vecinos_bloq

        acciones_ordenadas = sorted(
            acciones, key=dificultad_celda, reverse=True
        )
        return acciones_ordenadas[: self.max_candidatas_fruta]

    def _is_cutoff(self, estado: EstadoSnake, d: int) -> bool:
        """Indica si la búsqueda debe cortarse en este estado."""
        if self.game.es_terminal(estado):
            return True
        return d >= self.max_lim

    def _evaluar(self, estado: EstadoSnake) -> float:
        """Evalúa un estado desde la perspectiva de la serpiente, MAX.

        Premia la longitud y los grados de libertad de la cabeza, y
        penaliza el riesgo local, la distancia a la fruta y que la fruta
        esté encerrada. En estados terminales devuelve la utilidad del
        juego.
        """
        self.nodos_evaluados_minimax += 1
        if self.game.es_terminal(estado):
            util = self.game.utilidad(estado, "serpiente")
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
                if (
                    not (0 <= nx < estado.tamano and 0 <= ny < estado.tamano)
                    or (nx, ny) in ocupadas
                ):
                    encierro_fruta += 1.0

        # MAX quiere longitud y libertad altas, poca distancia a la
        # fruta y la fruta en una zona despejada.
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
        """Búsqueda H-Minimax con poda alfa-beta.

        Parameters
        ----------
        estado : EstadoSnake
            Estado desde el que se busca.
        d : int
            Profundidad actual.
        alfa, beta : float
            Cotas de la poda alfa-beta.

        Returns
        -------
        tuple
            ``(valor, mejor_accion)``.
        """
        jugador_turno = self.game.player(estado)
        if self._is_cutoff(estado, d):
            return self._evaluar(estado), None

        acciones = self._acciones_acotadas(estado)
        if not acciones:
            return self._evaluar(estado), None

        if jugador_turno == "serpiente":  # Jugador MAX.
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

        elif jugador_turno == "fruta":  # Jugador MIN, la fruta adversaria.
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
