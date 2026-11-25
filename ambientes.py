"""
Módulo ambientes.py
Contiene la formulación del entorno Snake y sus variantes siguiendo la
interfaz de los talleres del curso:
1. EstadoSnake: representación inmutable s = (C, f).
2. SubproblemaSnake: espacio de estados para alcanzar una fruta o la cola
   (compatible con nodo_hijo y los algoritmos de busqueda.py).
3. Snake: entorno secuencial completo (compatible con Episode y Experiment de utils.py).
4. SnakeJuegoAdversario: juego bipersonal de suma cero para la variante de
   fruta adversaria con Minimax y poda Alfa-Beta.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
from time import sleep
from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

Celda = Tuple[int, int]

# Direcciones consistentes con Laberinto (ambientes.py) y GoalBased (agentes.py)
DIRECCIONES: Dict[str, Tuple[int, int]] = {
    'norte': (0, 1),
    'sur': (0, -1),
    'este': (1, 0),
    'oeste': (-1, 0),
}


@dataclass(frozen=True)
class EstadoSnake:
    """
    Estado formal s = (C, f) de Snake.
    - cuerpo: tupla ordenada (c_1, ..., c_L) donde c_1 es la cabeza y c_L la cola.
    - fruta: coordenada (x, y) de la fruta actual (o tupla de frutas en variante múltiple).
    - tamano: dimensión N de la cuadrícula N x N (por defecto 10).
    - toroidal: True si el tablero no tiene bordes (variante toroidal).
    - obstaculos: tupla de celdas (x, y) con muros fijos.
    - turno: 'serpiente' o 'fruta' (usado solo en la variante adversaria Minimax).
    """
    cuerpo: Tuple[Celda, ...]
    fruta: Optional[Celda]
    tamano: int = 10
    toroidal: bool = False
    obstaculos: Tuple[Celda, ...] = ()
    frutas_extra: Tuple[Celda, ...] = ()
    turno: str = 'serpiente'

    @property
    def cabeza(self) -> Celda:
        return self.cuerpo[0]

    @property
    def cola(self) -> Celda:
        return self.cuerpo[-1]

    @property
    def longitud(self) -> int:
        return len(self.cuerpo)

    @property
    def todas_las_frutas(self) -> Tuple[Celda, ...]:
        if self.fruta is None:
            return self.frutas_extra
        return (self.fruta,) + self.frutas_extra

    def __str__(self) -> str:
        return f"EstadoSnake(cabeza={self.cabeza}, L={self.longitud}, fruta={self.fruta})"


# =============================================================================
# FUNCIONES AUXILIARES DE DINÁMICA Y RIESGO (MODELO DE TRANSICIÓN Y PÉRDIDA)
# =============================================================================

def mover_coordenada(
    celda: Celda, accion: str, tamano: int = 10, toroidal: bool = False
) -> Optional[Celda]:
    """Calcula la nueva celda h' = c_1 + delta(a) según bordes o toroide."""
    if accion not in DIRECCIONES:
        raise ValueError(f"Acción inválida: {accion}. Use {list(DIRECCIONES.keys())}")
    dx, dy = DIRECCIONES[accion]
    nx, ny = celda[0] + dx, celda[1] + dy
    if toroidal:
        return (nx % tamano, ny % tamano)
    if 0 <= nx < tamano and 0 <= ny < tamano:
        return (nx, ny)
    return None


def obtener_acciones_aplicables(estado: EstadoSnake) -> List[str]:
    """
    Determina las acciones aplicables en el estado s = (C, f).
    - No puede salir del tablero (salvo modo toroidal) ni chocar con obstáculos.
    - Para L >= 2 se excluye el giro de 180 grados (ir hacia c_2).
    - La celda de la cola c_L sí es válida porque se desplaza en el mismo turno.
    """
    acciones: List[str] = []
    cabeza = estado.cabeza
    cuello = estado.cuerpo[1] if estado.longitud >= 2 else None
    # El segmento c_L se libera al avanzar, por lo que solo bloquean c_1 ... c_{L-1}
    cuerpo_bloqueado = set(estado.cuerpo[:-1])
    obstaculos = set(estado.obstaculos)

    for acc in DIRECCIONES:
        nueva_h = mover_coordenada(cabeza, acc, estado.tamano, estado.toroidal)
        if nueva_h is None:
            continue
        if cuello is not None and nueva_h == cuello:
            continue
        if nueva_h in obstaculos or nueva_h in cuerpo_bloqueado:
            continue
        acciones.append(acc)
    return acciones


def aplicar_transicion(
    estado: EstadoSnake, accion: str, comer_fruta: bool = True
) -> EstadoSnake:
    """
    Modelo de transición determinista T(s, a) para un subproblema.
    Si h' coincide con una fruta y comer_fruta=True, la serpiente crece en 1 unidad.
    En caso contrario, el cuerpo avanza y la cola se recorre.
    """
    nueva_h = mover_coordenada(estado.cabeza, accion, estado.tamano, estado.toroidal)
    if nueva_h is None:
        raise ValueError(f"La acción {accion} sale del tablero desde {estado.cabeza}.")

    frutas_actuales = set(estado.todas_las_frutas)
    comio = comer_fruta and (nueva_h in frutas_actuales)

    if comio:
        nuevo_cuerpo = (nueva_h,) + estado.cuerpo
    else:
        nuevo_cuerpo = (nueva_h,) + estado.cuerpo[:-1]

    return EstadoSnake(
        cuerpo=nuevo_cuerpo,
        fruta=estado.fruta,
        tamano=estado.tamano,
        toroidal=estado.toroidal,
        obstaculos=estado.obstaculos,
        frutas_extra=estado.frutas_extra,
        turno=estado.turno,
    )


def calcular_riesgo_celda(estado: EstadoSnake) -> float:
    """
    Evalúa la función de riesgo r(h') en [0, 1] para la cabeza actual del estado.
    Cuenta cuántos de los 3 vecinos posibles hacia adelante están bloqueados por
    el cuerpo, muros u orillas del tablero.
    """
    cabeza = estado.cabeza
    cuello = estado.cuerpo[1] if estado.longitud >= 2 else None
    bloqueadas = set(estado.cuerpo[:-1]) | set(estado.obstaculos)

    vecinos_peligrosos = 0
    for acc in DIRECCIONES:
        vecino = mover_coordenada(cabeza, acc, estado.tamano, estado.toroidal)
        if vecino is None:
            vecinos_peligrosos += 1
        elif vecino == cuello:
            continue
        elif vecino in bloqueadas:
            vecinos_peligrosos += 1

    return vecinos_peligrosos / 3.0


# =============================================================================
# 1. CLASE ESPACIO DE ESTADOS (COMPATIBLE CON busqueda.py Y utils_espacios.py)
# =============================================================================

class SubproblemaSnake:
    """
    Representa el subproblema determinista de llevar la cabeza de la serpiente
    hasta una celda objetivo (la fruta actual o la cola en el chequeo de seguridad).
    Implementa exactamente los métodos exigidos por nodo_hijo() en busqueda.py:
    estado_inicial, acciones_aplicables, transicion, test_objetivo, costo y codigo.
    """

    def __init__(
        self,
        estado_inicial: Optional[EstadoSnake] = None,
        objetivo: Optional[Celda] = None,
        lambda_riesgo: float = 0.0,
        comer_en_objetivo: bool = True,
        codigo_cuerpo_completo: bool = False,
    ):
        if estado_inicial is None:
            estado_inicial = EstadoSnake(cuerpo=((5, 5),), fruta=(8, 8))
        self.estado_inicial = estado_inicial
        self.objetivo = objetivo if objetivo is not None else estado_inicial.fruta
        self.lambda_riesgo = lambda_riesgo
        self.comer_en_objetivo = comer_en_objetivo
        self.codigo_cuerpo_completo = codigo_cuerpo_completo

    def acciones_aplicables(self, estado: EstadoSnake) -> List[str]:
        """Devuelve la lista de acciones aplicables desde 'estado'."""
        return obtener_acciones_aplicables(estado)

    def transicion(self, estado: EstadoSnake, accion: str) -> EstadoSnake:
        """Devuelve el estado siguiente tras aplicar 'accion'."""
        nueva_h = mover_coordenada(estado.cabeza, accion, estado.tamano, estado.toroidal)
        es_meta = (self.objetivo is not None) and (nueva_h == self.objetivo)
        return aplicar_transicion(
            estado, accion, comer_fruta=(self.comer_en_objetivo and es_meta)
        )

    def test_objetivo(self, estado: EstadoSnake) -> bool:
        """Verifica si la cabeza llegó al objetivo del subproblema."""
        if estado.frutas_extra and self.objetivo == estado.fruta:
            return estado.cabeza in estado.todas_las_frutas
        return self.objetivo is not None and estado.cabeza == self.objetivo

    def costo(self, estado: EstadoSnake, accion: str) -> float:
        """
        Función de pérdida instantánea por paso:
        c(s, a, s') = 1 + lambda * r(h').
        Con lambda = 0 devuelve costo unitario 1.
        """
        if self.lambda_riesgo <= 0.0:
            return 1.0
        estado_sig = self.transicion(estado, accion)
        return 1.0 + self.lambda_riesgo * calcular_riesgo_celda(estado_sig)

    def codigo(self, estado: EstadoSnake) -> str:
        """
        Código único del estado para detección de ciclos y tabla de explorados.
        Por defecto codifica la celda de la cabeza 'x-y' para búsqueda en grafo eficiente.
        """
        if self.codigo_cuerpo_completo:
            return "|".join(f"{x}-{y}" for x, y in estado.cuerpo)
        x, y = estado.cabeza
        return f"{x}-{y}"

    def heuristica(self, estado: EstadoSnake) -> float:
        """
        Heurística por defecto h_1(s): distancia Manhattan (o toroidal si aplica).
        Permite invocar avara(problema, problema.heuristica) igual que en el Taller 10.
        """
        objetivos = (
            estado.todas_las_frutas
            if (estado.frutas_extra and self.objetivo == estado.fruta)
            else ((self.objetivo,) if self.objetivo is not None else ())
        )
        if not objetivos:
            return 0.0
        xh, yh = estado.cabeza
        distancias = []
        for ox, oy in objetivos:
            dx = abs(xh - ox)
            dy = abs(yh - oy)
            if estado.toroidal:
                dx = min(dx, estado.tamano - dx)
                dy = min(dy, estado.tamano - dy)
            distancias.append(float(dx + dy))
        return min(distancias)

    def render(self, estado: Optional[EstadoSnake] = None, ax=None):
        """Dibuja el estado del subproblema."""
        est = estado if estado is not None else self.estado_inicial
        return dibujar_tablero_snake(est, ax=ax)

    def pintar_camino(self, camino: List[str], sleep_time: float = 0.25):
        """Anima una secuencia de acciones sobre el subproblema (al estilo Rompecabezas)."""
        estados = [self.estado_inicial]
        for accion in camino:
            estados.append(self.transicion(estados[-1], accion))
        for est in estados:
            clear_output(wait=True)
            self.render(est)
            plt.show()
            sleep(sleep_time)


# =============================================================================
# 2. ENTORNO SECUENCIAL COMPLETO (COMPATIBLE CON Episode Y Experiment DE utils.py)
# =============================================================================

class Snake:
    """
    Entorno completo del juego Snake por turnos.
    Cumple la interfaz de Laberinto (Taller 10) y Episode (utils.py):
    - reset() -> devuelve el estado inicial.
    - step(accion) -> devuelve (new_state, reward, done).
    - test_objetivo() -> True si se llenó el tablero (L == N^2 - obstaculos).
    - render() -> visualiza el tablero con matplotlib.
    """

    def __init__(
        self,
        tamano: int = 10,
        semilla: int = 0,
        pos_inicial: Optional[Celda] = None,
        fruta_inicial: Optional[Celda] = None,
        toroidal: bool = False,
        num_obstaculos: int = 0,
        num_frutas: int = 1,
        lambda_riesgo: float = 0.0,
        k_max: int = 200,
        ganancia_fruta: float = 100.0,
        ganancia_victoria: float = 1000.0,
        perdida_choque: float = 100.0,
    ):
        self.tamano = tamano
        self.semilla = semilla
        self.pos_inicial_config = pos_inicial
        self.fruta_inicial_config = fruta_inicial
        self.toroidal = toroidal
        self.num_obstaculos = num_obstaculos
        self.num_frutas = max(1, num_frutas)
        self.lambda_riesgo = lambda_riesgo
        self.k_max = k_max

        # Parámetros de la función de Ganancia y Pérdida (Recompensa R)
        self.ganancia_fruta = ganancia_fruta
        self.ganancia_victoria = ganancia_victoria
        self.perdida_choque = perdida_choque

        self.rng = np.random.default_rng(semilla)
        self.estado: Optional[EstadoSnake] = None
        self.pasos: int = 0
        self.pasos_sin_comer: int = 0
        self.frutas_comidas: int = 0
        self.ganancia_total: float = 0.0
        self.perdida_total: float = 0.0
        self.done: bool = False
        self.causa_fin: str = "En curso"

        self.reset()

    def reset(self, semilla: Optional[int] = None) -> EstadoSnake:
        """
        Reinicia el entorno con la semilla dada para garantizar reproducibilidad
        exacta entre los distintos algoritmos.
        """
        if semilla is not None:
            self.semilla = semilla
        self.rng = np.random.default_rng(self.semilla)

        if self.pos_inicial_config is not None:
            cabeza_ini = self.pos_inicial_config
        else:
            cabeza_ini = (self.tamano // 2, self.tamano // 2)

        cuerpo_ini = (cabeza_ini,)

        # Generación reproducible de obstáculos fijos (si la variante está activa)
        obstaculos: Tuple[Celda, ...] = ()
        if self.num_obstaculos > 0:
            candidatas = [
                (x, y)
                for x in range(self.tamano)
                for y in range(self.tamano)
                if (x, y) != cabeza_ini
            ]
            indices = self.rng.choice(
                len(candidatas),
                size=min(self.num_obstaculos, len(candidatas) - 2),
                replace=False,
            )
            obstaculos = tuple(candidatas[int(i)] for i in indices)

        # Generación de fruta(s)
        ocupadas = set(cuerpo_ini) | set(obstaculos)
        frutas_generadas: List[Celda] = []
        if (
            self.fruta_inicial_config is not None
            and self.fruta_inicial_config not in ocupadas
        ):
            frutas_generadas.append(self.fruta_inicial_config)
            ocupadas.add(self.fruta_inicial_config)

        while len(frutas_generadas) < self.num_frutas:
            nueva_f = self._sortear_celda_libre(ocupadas)
            if nueva_f is None:
                break
            frutas_generadas.append(nueva_f)
            ocupadas.add(nueva_f)

        fruta_principal = frutas_generadas[0] if frutas_generadas else None
        frutas_extra = tuple(frutas_generadas[1:]) if len(frutas_generadas) > 1 else ()

        self.estado = EstadoSnake(
            cuerpo=cuerpo_ini,
            fruta=fruta_principal,
            tamano=self.tamano,
            toroidal=self.toroidal,
            obstaculos=obstaculos,
            frutas_extra=frutas_extra,
        )

        self.pasos = 0
        self.pasos_sin_comer = 0
        self.frutas_comidas = 0
        self.ganancia_total = 0.0
        self.perdida_total = 0.0
        self.done = False
        self.causa_fin = "En curso"
        return self.estado

    def _sortear_celda_libre(self, ocupadas: Set[Celda]) -> Optional[Celda]:
        """Elige una celda libre con distribución uniforme usando el generador con semilla."""
        libres = [
            (x, y)
            for x in range(self.tamano)
            for y in range(self.tamano)
            if (x, y) not in ocupadas
        ]
        if not libres:
            return None
        idx = int(self.rng.integers(0, len(libres)))
        return libres[idx]

    def acciones_aplicables(self, estado: Optional[EstadoSnake] = None) -> List[str]:
        """Retorna las acciones válidas en el estado dado (o en el estado actual)."""
        est = estado if estado is not None else self.estado
        if est is None:
            return []
        return obtener_acciones_aplicables(est)

    def test_objetivo(self, estado: Optional[EstadoSnake] = None) -> bool:
        """Condición de victoria de la partida: llenar todas las celdas disponibles."""
        est = estado if estado is not None else self.estado
        if est is None:
            return False
        meta = (est.tamano * est.tamano) - len(est.obstaculos)
        return est.longitud >= meta

    def step(self, accion: str) -> Tuple[EstadoSnake, float, bool]:
        """
        Ejecuta una acción en el entorno y retorna exactamente (new_state, reward, done)
        para compatibilidad directa con la clase Episode de utils.py.
        """
        if self.done or self.estado is None:
            return self.estado, 0.0, True

        acciones_validas = self.acciones_aplicables(self.estado)
        if accion not in acciones_validas:
            self.done = True
            self.causa_fin = f"Colisión / Acción inválida ({accion})"
            self.perdida_total += self.perdida_choque
            reward = -self.perdida_choque
            return self.estado, reward, True

        # Aplicamos transición física del cuerpo
        est_trans = aplicar_transicion(self.estado, accion, comer_fruta=True)
        perdida_paso = 1.0 + (
            self.lambda_riesgo * calcular_riesgo_celda(est_trans)
            if self.lambda_riesgo > 0.0
            else 0.0
        )

        self.pasos += 1
        self.perdida_total += perdida_paso

        frutas_previas = list(self.estado.todas_las_frutas)
        comio = est_trans.cabeza in frutas_previas
        ganancia_paso = 0.0

        if comio:
            self.frutas_comidas += 1
            self.pasos_sin_comer = 0
            frutas_restantes = [f for f in frutas_previas if f != est_trans.cabeza]

            if self.test_objetivo(est_trans):
                self.done = True
                self.causa_fin = "Victoria (Tablero lleno)"
                ganancia_paso = self.ganancia_victoria
                nueva_fruta = None
                nuevas_extra = ()
            else:
                ganancia_paso = self.ganancia_fruta
                ocupadas = (
                    set(est_trans.cuerpo)
                    | set(est_trans.obstaculos)
                    | set(frutas_restantes)
                )
                while len(frutas_restantes) < self.num_frutas:
                    nf = self._sortear_celda_libre(ocupadas)
                    if nf is None:
                        break
                    frutas_restantes.append(nf)
                    ocupadas.add(nf)
                nueva_fruta = frutas_restantes[0] if frutas_restantes else None
                nuevas_extra = (
                    tuple(frutas_restantes[1:]) if len(frutas_restantes) > 1 else ()
                )
        else:
            self.pasos_sin_comer += 1
            nueva_fruta = self.estado.fruta
            nuevas_extra = self.estado.frutas_extra

        self.ganancia_total += ganancia_paso
        reward = ganancia_paso - perdida_paso

        self.estado = EstadoSnake(
            cuerpo=est_trans.cuerpo,
            fruta=nueva_fruta,
            tamano=self.tamano,
            toroidal=self.toroidal,
            obstaculos=est_trans.obstaculos,
            frutas_extra=nuevas_extra,
        )

        # Verificación de estados terminales tras mover
        if not self.done:
            if self.pasos_sin_comer >= self.k_max:
                self.done = True
                self.causa_fin = f"Límite de pasos sin comer alcanzado (k_max={self.k_max})"
                self.perdida_total += self.perdida_choque
                reward -= self.perdida_choque
            elif len(self.acciones_aplicables(self.estado)) == 0:
                self.done = True
                self.causa_fin = "Encerrado (Sin acciones aplicables)"
                self.perdida_total += self.perdida_choque
                reward -= self.perdida_choque

        return self.estado, reward, self.done

    def render(
        self,
        estado: Optional[EstadoSnake] = None,
        camino: Optional[List[Celda]] = None,
        frontera: Optional[List[Celda]] = None,
        expandidos: Optional[List[Celda]] = None,
        celda_actual: Optional[Celda] = None,
        titulo: Optional[str] = None,
        ax=None,
    ):
        """Dibuja el tablero actual o cualquier estado junto con la búsqueda."""
        est = estado if estado is not None else self.estado
        if titulo is None and est is not None:
            titulo = (
                f"Longitud: {est.longitud} | Frutas: {self.frutas_comidas} | "
                f"Pasos: {self.pasos} | Estado: {self.causa_fin}"
            )
        return dibujar_tablero_snake(
            est,
            camino=camino,
            frontera=frontera,
            expandidos=expandidos,
            celda_actual=celda_actual,
            titulo=titulo,
            ax=ax,
        )


# =============================================================================
# 3. JUEGO ADVERSARIO (COMPATIBLE CON juegos.py Y EnvfromGameAndPl2 DE utils.py)
# =============================================================================

class SnakeJuegoAdversario:
    """
    Variante de la Sección 5: Fruta Adversaria como juego de suma cero.
    - Jugador MAX ('serpiente'): mueve la cabeza buscando sobrevivir y comer.
    - Jugador MIN ('fruta'): cuando la serpiente come, coloca la nueva fruta en la
      celda libre que minimice la utilidad de la serpiente (Minimax con poda Alfa-Beta).
    Implementa la interfaz de juegos.py (player, acciones, resultado, es_terminal, utilidad, render).
    """

    def __init__(self, tamano: int = 10, obstaculos: Tuple[Celda, ...] = ()):
        mitad = tamano // 2
        self.tamano = tamano
        self.estado_inicial = EstadoSnake(
            cuerpo=((mitad, mitad),),
            fruta=(mitad + 2, mitad),
            tamano=tamano,
            obstaculos=obstaculos,
            turno='serpiente',
        )

    def player(self, estado: EstadoSnake) -> str:
        """Devuelve quién tiene el turno: 'serpiente' (MAX) o 'fruta' (MIN)."""
        return estado.turno

    def acciones(self, estado: EstadoSnake) -> List:
        """
        Si juega 'serpiente', devuelve las direcciones válidas.
        Si juega 'fruta', devuelve las celdas libres donde puede colocar la manzana.
        """
        if estado.turno == 'serpiente':
            return obtener_acciones_aplicables(estado)
        ocupadas = set(estado.cuerpo) | set(estado.obstaculos)
        libres = [
            (x, y)
            for x in range(estado.tamano)
            for y in range(estado.tamano)
            if (x, y) not in ocupadas
        ]
        return libres

    def resultado(self, estado: EstadoSnake, accion) -> EstadoSnake:
        """Aplica la jugada de la serpiente o la colocación de la fruta adversaria."""
        if estado.turno == 'serpiente':
            est_sig = aplicar_transicion(estado, accion, comer_fruta=True)
            comio = (estado.fruta is not None) and (est_sig.cabeza == estado.fruta)
            if comio:
                # Si comió, el siguiente turno es del adversario para colocar la fruta
                return EstadoSnake(
                    cuerpo=est_sig.cuerpo,
                    fruta=None,
                    tamano=estado.tamano,
                    toroidal=estado.toroidal,
                    obstaculos=estado.obstaculos,
                    turno='fruta',
                )
            return EstadoSnake(
                cuerpo=est_sig.cuerpo,
                fruta=estado.fruta,
                tamano=estado.tamano,
                toroidal=estado.toroidal,
                obstaculos=estado.obstaculos,
                turno='serpiente',
            )
        else:
            # Jugada del oponente: accion es la tupla (x, y) donde coloca la fruta
            return EstadoSnake(
                cuerpo=estado.cuerpo,
                fruta=accion,
                tamano=estado.tamano,
                toroidal=estado.toroidal,
                obstaculos=estado.obstaculos,
                turno='serpiente',
            )

    def es_terminal(self, estado: EstadoSnake) -> bool:
        """El juego termina si el tablero se llena o si el jugador en turno no tiene jugadas."""
        meta = (estado.tamano * estado.tamano) - len(estado.obstaculos)
        if estado.longitud >= meta:
            return True
        if estado.turno == 'serpiente' and len(obtener_acciones_aplicables(estado)) == 0:
            return True
        if estado.turno == 'fruta' and len(self.acciones(estado)) == 0:
            return True
        return False

    def utilidad(self, estado: EstadoSnake, jugador: str = 'serpiente') -> Optional[float]:
        """Devuelve +1000 si gana la serpiente, -1000 si queda atrapada."""
        if not self.es_terminal(estado):
            return None
        meta = (estado.tamano * estado.tamano) - len(estado.obstaculos)
        valor = 1000.0 if estado.longitud >= meta else -1000.0
        return valor if jugador == 'serpiente' else -valor

    def render(self, estado: EstadoSnake):
        dibujar_tablero_snake(estado, titulo=f"Modo Adversario | Turno: {estado.turno}")


# =============================================================================
# 4. FUNCIÓN DE RENDERIZADO CON MATPLOTLIB PATCHES (ESTILO TALLERES DEL CURSO)
# =============================================================================

def dibujar_tablero_snake(
    estado: EstadoSnake,
    camino: Optional[List[Celda]] = None,
    frontera: Optional[List[Celda]] = None,
    expandidos: Optional[List[Celda]] = None,
    celda_actual: Optional[Celda] = None,
    titulo: Optional[str] = None,
    ax=None,
):
    """
    Dibuja la cuadrícula de Snake usando matplotlib.patches.Rectangle,
    diferenciando visualmente:
    - Nodos expandidos (azul claro)
    - Frontera actual (amarillo)
    - Nodo actual de búsqueda (naranja)
    - Camino solución planeado (verde claro)
    - Cuerpo de la serpiente con degradado desde la cabeza c_1 hasta la cola c_L
    - Fruta(s) y obstáculos fijos.
    """
    mostrar_plt = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        ax.clear()

    n = estado.tamano
    step = 1.0 / n

    # Fondo del tablero (mismo estilo cornsilk de Laberinto y OchoReinas)
    ax.add_patch(
        patches.Rectangle(
            (0, 0),
            1.0,
            1.0,
            facecolor='cornsilk',
            edgecolor='black',
            linewidth=2,
        )
    )

    # 1. Celdas expandidas en el proceso de búsqueda
    if expandidos:
        for x, y in expandidos:
            ax.add_patch(
                patches.Rectangle(
                    (x * step, y * step),
                    step,
                    step,
                    facecolor='#b3cde3',
                    alpha=0.65,
                )
            )

    # 2. Celdas en la frontera de búsqueda
    if frontera:
        for x, y in frontera:
            ax.add_patch(
                patches.Rectangle(
                    (x * step, y * step),
                    step,
                    step,
                    facecolor='#fed976',
                    alpha=0.85,
                )
            )

    # 3. Camino planeado hacia la meta
    if camino:
        for x, y in camino:
            ax.add_patch(
                patches.Rectangle(
                    (x * step + step * 0.18, y * step + step * 0.18),
                    step * 0.64,
                    step * 0.64,
                    facecolor='#78c679',
                    alpha=0.75,
                )
            )

    # 4. Celda actual que está evaluando el algoritmo
    if celda_actual is not None:
        cx, cy = celda_actual
        ax.add_patch(
            patches.Rectangle(
                (cx * step, cy * step),
                step,
                step,
                facecolor='#fd8d3c',
                edgecolor='red',
                linewidth=2,
                alpha=0.9,
            )
        )

    # 5. Obstáculos fijos (muros)
    for ox, oy in estado.obstaculos:
        ax.add_patch(
            patches.Rectangle(
                (ox * step, oy * step),
                step,
                step,
                facecolor='#252525',
            )
        )

    # 6. Cuerpo de la serpiente (degradado de verde oscuro en cabeza a verde claro en cola)
    longitud = estado.longitud
    for idx, (sx, sy) in enumerate(estado.cuerpo):
        if idx == 0:
            color = '#005a32'  # Cabeza c_1
        elif idx == longitud - 1:
            color = '#a1d99b'  # Cola c_L
        else:
            frac = idx / max(1, longitud - 1)
            r = 0.05 + 0.45 * frac
            g = 0.45 + 0.35 * frac
            b = 0.20 + 0.30 * frac
            color = (r, g, b)

        margen = step * 0.06
        ax.add_patch(
            patches.Rectangle(
                (sx * step + margen, sy * step + margen),
                step - 2 * margen,
                step - 2 * margen,
                facecolor=color,
                edgecolor='black',
                linewidth=1.0,
            )
        )
        if idx == 0:
            ax.text(
                (sx + 0.5) * step,
                (sy + 0.5) * step,
                "H",
                color="white",
                weight="bold",
                ha="center",
                va="center",
                fontsize=max(7, int(110 / n)),
            )
        elif idx == longitud - 1 and longitud > 1:
            ax.text(
                (sx + 0.5) * step,
                (sy + 0.5) * step,
                "T",
                color="black",
                weight="bold",
                ha="center",
                va="center",
                fontsize=max(6, int(95 / n)),
            )

    # 7. Fruta(s)
    for fx, fy in estado.todas_las_frutas:
        circulo = patches.Circle(
            ((fx + 0.5) * step, (fy + 0.5) * step),
            radius=step * 0.36,
            facecolor='#e31a1c',
            edgecolor='darkred',
            linewidth=1.5,
        )
        ax.add_patch(circulo)

    # Líneas de la cuadrícula
    for j in range(1, n):
        loc = j * step
        ax.add_patch(patches.Rectangle((0, loc), 1.0, 0.003, facecolor='gray', alpha=0.4))
        ax.add_patch(patches.Rectangle((loc, 0), 0.003, 1.0, facecolor='gray', alpha=0.4))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    if titulo:
        if " | Fin:" in titulo and "\nFin:" not in titulo:
            titulo = titulo.replace(" | Fin:", "\nFin:")
        ax.set_title(
            titulo,
            fontsize=9.5,
            weight='bold',
            pad=10,
            linespacing=1.35,
            wrap=True,
            bbox=dict(
                boxstyle='round,pad=0.35',
                facecolor='#f8f9fa',
                edgecolor='#cccccc',
                alpha=0.9,
            ),
        )

    if mostrar_plt:
        plt.show()
    return ax