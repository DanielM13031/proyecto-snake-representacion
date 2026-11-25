"""Algoritmos de búsqueda en grafo y heurísticas para Snake.

El módulo incluye:

- Estructuras de búsqueda: ``ListaPrioritaria``, ``Nodo``,
  ``nodo_hijo``, ``solucion``, ``depth``, ``camino_codigos`` e
  ``is_cycle``.
- Heurísticas: ``heuristica_manhattan``, ``heuristica_euclidea`` y
  ``heuristica_riesgo``.
- Algoritmos no informados: ``anchura`` (BFS), ``profundidad`` (DFS),
  ``backtracking`` y ``dijkstra`` (UCS).
- Algoritmos informados: ``avara``, ``a_estrella`` (A*) y
  ``busqueda_haz`` (Beam Search).
- Registro de métricas y de la traza paso a paso que usa la interfaz
  para animar la frontera y los nodos expandidos.
"""

import heapq
import inspect
import itertools
import math
import time
from collections import deque, namedtuple
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from ambientes import (
    Celda,
    EstadoSnake,
    SubproblemaSnake,
    calcular_riesgo_celda,
)

# Cotas de seguridad: tamaño máximo de la frontera y profundidad máxima
# del backtracking. Son holgadas para tableros de 10 x 10 hasta 20 x 20.
max_size = 50000
max_size_nodo = 2500

# Pareja (elemento, prioridad) que devuelve ListaPrioritaria.
Tupla = namedtuple("Tupla", ["elemento", "valor"])


# =============================================================================
# 1. Estructuras de datos de la búsqueda
# =============================================================================


class ListaPrioritaria:
    """Cola de prioridad de mínimos implementada con ``heapq``.

    Un contador incremental desempata elementos con la misma prioridad,
    de modo que salen en el orden en que entraron y nunca se comparan
    los elementos entre sí.
    """

    def __init__(self):
        """Crea una cola vacía."""
        self._heap: List[Tuple[float, int, Any]] = []
        self._counter = itertools.count()

    def __len__(self) -> int:
        """Número de elementos en la cola."""
        return len(self._heap)

    def push(self, elemento: Any, valor: float) -> None:
        """Agrega ``elemento`` con prioridad ``valor``."""
        heapq.heappush(
            self._heap, (float(valor), next(self._counter), elemento)
        )

    def pop(self, with_value: bool = False) -> Any:
        """Saca el elemento de menor prioridad.

        Parameters
        ----------
        with_value : bool
            Si es verdadero, devuelve ``Tupla(elemento, valor)``.

        Raises
        ------
        IndexError
            Si la cola está vacía.
        """
        if not self._heap:
            raise IndexError("pop desde una ListaPrioritaria vacía")
        valor, _, elemento = heapq.heappop(self._heap)
        if with_value:
            return Tupla(elemento, valor)
        return elemento

    def remove(self, elemento: Any) -> None:
        """Elimina la primera aparición de ``elemento``, si existe."""
        for i, (_, _, el) in enumerate(self._heap):
            if el == elemento:
                self._heap.pop(i)
                heapq.heapify(self._heap)
                break

    def elementos(self) -> List[Any]:
        """Elementos de la cola ordenados por prioridad."""
        return [el for _, _, el in sorted(self._heap)]

    @property
    def pila(self) -> List[Tupla]:
        """Parejas ``(elemento, valor)`` ordenadas por prioridad."""
        return [Tupla(el, val) for val, _, el in sorted(self._heap)]

    def is_empty(self) -> bool:
        """Indica si la cola está vacía."""
        return len(self._heap) == 0

    def __str__(self) -> str:
        """Representación ``[(elemento,valor), ...]`` en orden."""
        elementos_ordenados = sorted(self._heap)
        interior = ", ".join(
            f"({el},{val})" for val, _, el in elementos_ordenados
        )
        return f"[{interior}]"


class Nodo:
    """Nodo del árbol de búsqueda.

    Parameters
    ----------
    estado : Any
        Estado que representa el nodo.
    madre : Nodo or None
        Nodo padre; ``None`` en la raíz.
    accion : str or None
        Acción que llevó de la madre a este nodo.
    costo_camino : float
        Costo acumulado g(n) desde la raíz.
    codigo : str
        Código del estado para la tabla de explorados.
    heuristica : float
        Valor h(n) de la heurística.
    valor_f : float, optional
        Prioridad f(n). Por defecto, g(n) + h(n).

    Notes
    -----
    En el nodo solución se guardan además las métricas de la búsqueda:
    nodos generados y expandidos, frontera máxima, tiempo y traza.
    """

    def __init__(
        self,
        estado: Any,
        madre: Optional["Nodo"],
        accion: Optional[str],
        costo_camino: float,
        codigo: str,
        heuristica: float = 0.0,
        valor_f: Optional[float] = None,
    ):
        """Crea el nodo y calcula su profundidad según la madre."""
        self.estado = estado
        self.madre = madre
        self.accion = accion
        self.costo_camino = float(costo_camino)
        self.codigo = codigo
        self.heuristica = float(heuristica)
        self.valor_f = (
            float(valor_f)
            if valor_f is not None
            else float(costo_camino + heuristica)
        )
        self._profundidad: int = (
            0 if madre is None else (madre._profundidad + 1)
        )

        # Métricas de la búsqueda; solo se llenan en el nodo solución.
        self.nodos_generados: int = 0
        self.nodos_expandidos: int = 0
        self.max_frontera: int = 0
        self.tiempo_ms: float = 0.0
        self.traza: List["PasoBusqueda"] = []


def nodo_hijo(problema: Any, madre: Nodo, accion: str) -> Nodo:
    """Genera el nodo hijo de aplicar ``accion`` a ``madre``.

    El costo del hijo es el de la madre más el costo del paso según el
    problema.
    """
    estado = problema.transicion(madre.estado, accion)
    costo_camino = madre.costo_camino + problema.costo(madre.estado, accion)
    codigo = problema.codigo(estado)
    return Nodo(estado, madre, accion, costo_camino, codigo)


def solucion(n: Optional[Nodo]) -> List[str]:
    """Devuelve las acciones desde la raíz hasta el nodo ``n``."""
    if n is None:
        return []
    acciones: List[str] = []
    actual: Optional[Nodo] = n
    while actual is not None and actual.madre is not None:
        acciones.append(actual.accion)
        actual = actual.madre
    acciones.reverse()
    return acciones


def camino_celdas(n: Optional[Nodo]) -> List[Celda]:
    """Devuelve las celdas que recorre la cabeza hasta el nodo ``n``."""
    if n is None:
        return []
    celdas: List[Celda] = []
    actual: Optional[Nodo] = n
    while actual is not None:
        if hasattr(actual.estado, "cabeza"):
            celdas.append(actual.estado.cabeza)
        actual = actual.madre
    celdas.reverse()
    return celdas


def depth(nodo: Optional[Nodo]) -> int:
    """Profundidad del nodo en el árbol de búsqueda."""
    if nodo is None:
        return 0
    return nodo._profundidad


def camino_codigos(nodo: Optional[Nodo]) -> List[str]:
    """Lista de códigos desde la raíz hasta el nodo actual."""
    if nodo is None:
        return []
    codigos: List[str] = []
    actual: Optional[Nodo] = nodo
    while actual is not None:
        codigos.append(actual.codigo)
        actual = actual.madre
    codigos.reverse()
    return codigos


def is_cycle(nodo: Nodo) -> bool:
    """Indica si el código del nodo ya aparece entre sus ancestros."""
    codigo_actual = nodo.codigo
    actual = nodo.madre
    while actual is not None:
        if actual.codigo == codigo_actual:
            return True
        actual = actual.madre
    return False


# =============================================================================
# 2. Heurísticas
# =============================================================================


def _obtener_objetivos(
    estado: EstadoSnake, objetivo: Optional[Celda] = None
) -> Tuple[Celda, ...]:
    """Devuelve las celdas objetivo para evaluar una heurística.

    Si el objetivo es la fruta principal y hay varias frutas, devuelve
    todas; si no se da objetivo, devuelve las frutas del estado.
    """
    if objetivo is not None:
        if estado.frutas_extra and objetivo == estado.fruta:
            return estado.todas_las_frutas
        return (objetivo,)
    return estado.todas_las_frutas


def heuristica_manhattan(
    estado: EstadoSnake, objetivo: Optional[Celda] = None
) -> float:
    """Heurística h_1: distancia Manhattan al objetivo más cercano.

    En el modo toroidal usa la distancia Manhattan toroidal. Es
    admisible y consistente porque cada paso mueve la cabeza una celda
    y cuesta al menos 1, con o sin costo por riesgo.
    """
    objetivos = _obtener_objetivos(estado, objetivo)
    if not objetivos:
        return 0.0
    xh, yh = estado.cabeza
    mejor = math.inf
    for ox, oy in objetivos:
        dx = abs(xh - ox)
        dy = abs(yh - oy)
        if estado.toroidal:
            dx = min(dx, estado.tamano - dx)
            dy = min(dy, estado.tamano - dy)
        d = float(dx + dy)
        if d < mejor:
            mejor = d
    return mejor


def heuristica_euclidea(
    estado: EstadoSnake, objetivo: Optional[Celda] = None
) -> float:
    """Heurística h_2: distancia euclídea al objetivo más cercano.

    Es admisible porque h_2(s) <= h_1(s) <= h*(s), pero informa menos
    que la distancia Manhattan.
    """
    objetivos = _obtener_objetivos(estado, objetivo)
    if not objetivos:
        return 0.0
    xh, yh = estado.cabeza
    mejor = math.inf
    for ox, oy in objetivos:
        dx = abs(xh - ox)
        dy = abs(yh - oy)
        if estado.toroidal:
            dx = min(dx, estado.tamano - dx)
            dy = min(dy, estado.tamano - dy)
        d = float(math.hypot(dx, dy))
        if d < mejor:
            mejor = d
    return mejor


def heuristica_riesgo(
    estado: EstadoSnake, objetivo: Optional[Celda] = None
) -> float:
    """Heurística h_3: distancia Manhattan más 2 * r(h).

    La penalización por vecinos bloqueados alrededor de la cabeza
    favorece zonas despejadas, pero puede sobreestimar el costo real,
    por lo que no es admisible.
    """
    base = heuristica_manhattan(estado, objetivo)
    return base + 2.0 * calcular_riesgo_celda(estado)


# Heurísticas disponibles por nombre, para elegirlas desde texto.
CATALOGO_HEURISTICAS: Dict[str, Callable] = {
    "manhattan": heuristica_manhattan,
    "euclidea": heuristica_euclidea,
    "riesgo": heuristica_riesgo,
}


def _evaluar_heuristica(
    fn_heuristica: Callable, problema: Any, estado: EstadoSnake
) -> float:
    """Evalúa una heurística con firma ``f(s)`` o ``f(s, objetivo)``.

    Si la función acepta dos parámetros y el problema tiene objetivo,
    se le pasa ese objetivo; en otro caso se llama solo con el estado.
    """
    if fn_heuristica is None:
        return 0.0
    try:
        params = inspect.signature(fn_heuristica).parameters
        if len(params) >= 2 and hasattr(problema, "objetivo"):
            return float(fn_heuristica(estado, problema.objetivo))
    except (ValueError, TypeError):
        pass
    return float(fn_heuristica(estado))


# =============================================================================
# 3. Traza y métricas de la búsqueda
# =============================================================================


@dataclass
class PasoBusqueda:
    """Estado interno del algoritmo en una iteración.

    Lo usa la interfaz para reproducir la búsqueda paso a paso: celda
    actual, valores g, h y f, frontera, celdas expandidas y camino
    parcial.
    """

    iteracion: int
    celda_actual: Celda
    accion_tomada: Optional[str]
    g: float
    h: float
    f: float
    profundidad: int
    tamano_frontera: int
    nodos_generados: int
    nodos_expandidos: int
    frontera_celdas: List[Celda]
    expandidos_celdas: List[Celda]
    camino_parcial: List[Celda]


@dataclass
class ResultadoBusqueda:
    """Resultado y métricas de una ejecución de búsqueda.

    Se guarda en ``problema.ultimo_resultado`` al terminar la búsqueda.
    """

    algoritmo: str
    heuristica: str
    exito: bool
    nodo_solucion: Optional[Nodo] = None
    acciones: List[str] = field(default_factory=list)
    camino_celdas: List[Celda] = field(default_factory=list)
    estado_final: Optional[EstadoSnake] = None
    costo_total: float = math.inf
    longitud_solucion: int = 0
    nodos_generados: int = 0
    nodos_expandidos: int = 0
    max_frontera: int = 0
    tiempo_ms: float = 0.0
    es_optimo: Optional[bool] = None
    traza: List[PasoBusqueda] = field(default_factory=list)


def _capturar_paso(
    traza: List[PasoBusqueda],
    iteracion: int,
    nodo: Nodo,
    tamano_frontera: int,
    nodos_generados: int,
    nodos_expandidos: int,
    nodos_frontera: Iterable[Nodo],
    celdas_expandidas: Set[Celda],
) -> None:
    """Agrega a ``traza`` el estado de la búsqueda en esta iteración."""
    celda_act = (
        nodo.estado.cabeza if hasattr(nodo.estado, "cabeza") else (0, 0)
    )
    frontera_coords = [
        n.estado.cabeza for n in nodos_frontera if hasattr(n.estado, "cabeza")
    ]
    traza.append(
        PasoBusqueda(
            iteracion=iteracion,
            celda_actual=celda_act,
            accion_tomada=nodo.accion,
            g=round(nodo.costo_camino, 3),
            h=round(nodo.heuristica, 3),
            f=round(nodo.valor_f, 3),
            profundidad=depth(nodo),
            tamano_frontera=tamano_frontera,
            nodos_generados=nodos_generados,
            nodos_expandidos=nodos_expandidos,
            frontera_celdas=list(dict.fromkeys(frontera_coords)),
            expandidos_celdas=list(celdas_expandidas),
            camino_parcial=camino_celdas(nodo),
        )
    )


def _finalizar_resultado(
    problema: Any,
    algoritmo: str,
    nombre_heur: str,
    nodo_sol: Optional[Nodo],
    nodos_generados: int,
    nodos_expandidos: int,
    max_frontera: int,
    t0: float,
    es_optimo: bool,
    traza: List[PasoBusqueda],
) -> Optional[Nodo]:
    """Registra las métricas finales de una búsqueda.

    Guarda un ``ResultadoBusqueda`` en ``problema.ultimo_resultado`` y,
    si hubo solución, copia las métricas en el nodo solución.

    Returns
    -------
    Nodo or None
        El nodo solución, o ``None`` si la búsqueda falló.
    """
    tiempo_ms = (time.perf_counter() - t0) * 1000.0
    if nodo_sol is not None:
        nodo_sol.nodos_generados = nodos_generados
        nodo_sol.nodos_expandidos = nodos_expandidos
        nodo_sol.max_frontera = max_frontera
        nodo_sol.tiempo_ms = tiempo_ms
        nodo_sol.traza = traza
        accs = solucion(nodo_sol)
        celdas = camino_celdas(nodo_sol)
        res = ResultadoBusqueda(
            algoritmo=algoritmo,
            heuristica=nombre_heur,
            exito=True,
            nodo_solucion=nodo_sol,
            acciones=accs,
            camino_celdas=celdas,
            estado_final=nodo_sol.estado,
            costo_total=nodo_sol.costo_camino,
            longitud_solucion=len(accs),
            nodos_generados=nodos_generados,
            nodos_expandidos=nodos_expandidos,
            max_frontera=max_frontera,
            tiempo_ms=tiempo_ms,
            es_optimo=es_optimo,
            traza=traza,
        )
    else:
        res = ResultadoBusqueda(
            algoritmo=algoritmo,
            heuristica=nombre_heur,
            exito=False,
            nodos_generados=nodos_generados,
            nodos_expandidos=nodos_expandidos,
            max_frontera=max_frontera,
            tiempo_ms=tiempo_ms,
            es_optimo=False,
            traza=traza,
        )
    problema.ultimo_resultado = res
    return nodo_sol


# =============================================================================
# 4. Algoritmos de búsqueda en grafo
# =============================================================================
#
# Todos los algoritmos reciben un problema con la interfaz de
# SubproblemaSnake, devuelven el nodo solución o None y guardan sus
# métricas en problema.ultimo_resultado. Con guardar_traza=True además
# registran cada iteración para la animación paso a paso.


def anchura(problema: Any, guardar_traza: bool = False) -> Optional[Nodo]:
    """Búsqueda primero en anchura (BFS) en grafo.

    Expande por niveles con una cola FIFO y prueba el objetivo al
    generar cada hijo. Encuentra el camino con menos pasos, que es el de
    menor costo solo cuando ``lambda_riesgo == 0``.
    """
    t0 = time.perf_counter()
    s0 = problema.estado_inicial
    cod0 = problema.codigo(s0)
    raiz = Nodo(s0, None, None, 0.0, cod0)

    nodos_generados = 1
    nodos_expandidos = 0
    max_frontera = 1
    traza: List[PasoBusqueda] = []
    celdas_expandidas: Set[Celda] = set()

    # Caso trivial: el estado inicial ya es objetivo.
    if problema.test_objetivo(raiz.estado):
        if guardar_traza and hasattr(raiz.estado, "cabeza"):
            celdas_expandidas.add(raiz.estado.cabeza)
            _capturar_paso(traza, 1, raiz, 0, 1, 1, [], celdas_expandidas)
        return _finalizar_resultado(
            problema, "BFS", "ninguna", raiz, 1, 0, 1, t0, True, traza
        )

    # Frontera FIFO; un estado se marca como explorado al generarlo.
    frontera: deque[Nodo] = deque([raiz])
    explorados: Set[str] = {cod0}

    while len(frontera) > 0:
        max_frontera = max(max_frontera, len(frontera))
        nodo = frontera.popleft()
        nodos_expandidos += 1
        if hasattr(nodo.estado, "cabeza"):
            celdas_expandidas.add(nodo.estado.cabeza)

        if guardar_traza:
            _capturar_paso(
                traza,
                nodos_expandidos,
                nodo,
                len(frontera),
                nodos_generados,
                nodos_expandidos,
                frontera,
                celdas_expandidas,
            )

        for a in problema.acciones_aplicables(nodo.estado):
            hijo = nodo_hijo(problema, nodo, a)
            if hijo.codigo in explorados:
                continue
            explorados.add(hijo.codigo)
            nodos_generados += 1

            # Prueba temprana del objetivo, al generar el hijo.
            if problema.test_objetivo(hijo.estado):
                if guardar_traza:
                    if hasattr(hijo.estado, "cabeza"):
                        celdas_expandidas.add(hijo.estado.cabeza)
                    _capturar_paso(
                        traza,
                        nodos_expandidos + 1,
                        hijo,
                        len(frontera),
                        nodos_generados,
                        nodos_expandidos,
                        frontera,
                        celdas_expandidas,
                    )
                # BFS solo garantiza el óptimo con costo unitario.
                opt = getattr(problema, "lambda_riesgo", 0.0) == 0.0
                return _finalizar_resultado(
                    problema,
                    "BFS",
                    "ninguna",
                    hijo,
                    nodos_generados,
                    nodos_expandidos,
                    max(max_frontera, len(frontera)),
                    t0,
                    opt,
                    traza,
                )
            frontera.append(hijo)
            assert len(frontera) < max_size

    return _finalizar_resultado(
        problema,
        "BFS",
        "ninguna",
        None,
        nodos_generados,
        nodos_expandidos,
        max_frontera,
        t0,
        False,
        traza,
    )


def profundidad(problema: Any, guardar_traza: bool = False) -> Optional[Nodo]:
    """Búsqueda primero en profundidad (DFS) en grafo.

    Usa una pila y sigue cada rama hasta el fondo antes de retroceder.
    Es completa en grafos finitos pero no óptima: suele encontrar
    caminos largos e irregulares.
    """
    t0 = time.perf_counter()
    s0 = problema.estado_inicial
    cod0 = problema.codigo(s0)
    raiz = Nodo(s0, None, None, 0.0, cod0)

    frontera: List[Nodo] = [raiz]
    explorados: Set[str] = set()
    celdas_expandidas: Set[Celda] = set()
    traza: List[PasoBusqueda] = []

    nodos_generados = 1
    nodos_expandidos = 0
    max_frontera = 1

    while len(frontera) > 0:
        max_frontera = max(max_frontera, len(frontera))
        nodo = frontera.pop()
        # Un estado se marca como explorado al sacarlo de la pila, por
        # lo que se descartan las copias que llegaron antes por otra
        # rama.
        if nodo.codigo in explorados:
            continue
        explorados.add(nodo.codigo)

        nodos_expandidos += 1
        if hasattr(nodo.estado, "cabeza"):
            celdas_expandidas.add(nodo.estado.cabeza)

        if guardar_traza:
            _capturar_paso(
                traza,
                nodos_expandidos,
                nodo,
                len(frontera),
                nodos_generados,
                nodos_expandidos,
                frontera,
                celdas_expandidas,
            )

        if problema.test_objetivo(nodo.estado):
            return _finalizar_resultado(
                problema,
                "DFS",
                "ninguna",
                nodo,
                nodos_generados,
                nodos_expandidos,
                max_frontera,
                t0,
                False,
                traza,
            )

        # Se apilan en orden inverso para expandir primero la primera
        # acción aplicable.
        acciones = problema.acciones_aplicables(nodo.estado)
        for a in reversed(acciones):
            hijo = nodo_hijo(problema, nodo, a)
            if hijo.codigo not in explorados:
                nodos_generados += 1
                frontera.append(hijo)
                assert len(frontera) < max_size

    return _finalizar_resultado(
        problema,
        "DFS",
        "ninguna",
        None,
        nodos_generados,
        nodos_expandidos,
        max_frontera,
        t0,
        False,
        traza,
    )


def backtracking(problema: Any, nodo: Optional[Nodo] = None) -> Optional[Nodo]:
    """Búsqueda por backtracking recursivo.

    Recorre las ramas en profundidad evitando ciclos dentro de cada
    rama. No registra métricas; se conserva como referencia.
    """
    if nodo is None:
        s0 = problema.estado_inicial
        nodo = Nodo(s0, None, None, 0.0, problema.codigo(s0))
    if problema.test_objetivo(nodo.estado):
        return nodo
    assert depth(nodo) < max_size_nodo
    for a in problema.acciones_aplicables(nodo.estado):
        hijo = nodo_hijo(problema, nodo, a)
        if not is_cycle(hijo):
            resultado = backtracking(problema, hijo)
            if resultado is not None:
                return resultado
    return None


def dijkstra(problema: Any, guardar_traza: bool = False) -> Optional[Nodo]:
    """Búsqueda de costo uniforme (UCS, Dijkstra) en grafo.

    Expande siempre el nodo de menor costo acumulado g(n). Es completa y
    óptima para cualquier costo por paso positivo.
    """
    t0 = time.perf_counter()
    s = problema.estado_inicial
    cod = problema.codigo(s)
    nodo = Nodo(s, None, None, 0.0, cod, heuristica=0.0, valor_f=0.0)

    frontera = ListaPrioritaria()
    frontera.push(nodo, 0.0)
    explorados: Dict[str, float] = {cod: 0.0}
    celdas_expandidas: Set[Celda] = set()
    traza: List[PasoBusqueda] = []

    nodos_generados = 1
    nodos_expandidos = 0
    max_frontera = 1

    while not frontera.is_empty():
        max_frontera = max(max_frontera, len(frontera))
        nodo = frontera.pop()

        # Se descartan copias obsoletas: ya se encontró un camino más
        # barato a este estado.
        if nodo.costo_camino > explorados.get(nodo.codigo, math.inf) + 1e-9:
            continue

        nodos_expandidos += 1
        if hasattr(nodo.estado, "cabeza"):
            celdas_expandidas.add(nodo.estado.cabeza)

        if guardar_traza:
            _capturar_paso(
                traza,
                nodos_expandidos,
                nodo,
                len(frontera),
                nodos_generados,
                nodos_expandidos,
                frontera.elementos(),
                celdas_expandidas,
            )

        # La prueba del objetivo se hace al expandir para garantizar el
        # óptimo.
        if problema.test_objetivo(nodo.estado):
            return _finalizar_resultado(
                problema,
                "UCS",
                "ninguna",
                nodo,
                nodos_generados,
                nodos_expandidos,
                max_frontera,
                t0,
                True,
                traza,
            )

        for a in problema.acciones_aplicables(nodo.estado):
            hijo = nodo_hijo(problema, nodo, a)
            cod = hijo.codigo
            c = hijo.costo_camino
            hijo.valor_f = c
            # Solo se agrega si el estado es nuevo o el camino es mejor.
            if (cod not in explorados) or (c + 1e-9 < explorados[cod]):
                explorados[cod] = c
                nodos_generados += 1
                frontera.push(hijo, c)
                assert len(frontera) < max_size

    return _finalizar_resultado(
        problema,
        "UCS",
        "ninguna",
        None,
        nodos_generados,
        nodos_expandidos,
        max_frontera,
        t0,
        False,
        traza,
    )


def avara(
    problema: Any,
    heuristica: Optional[Callable] = None,
    guardar_traza: bool = False,
) -> Optional[Nodo]:
    """Búsqueda avara (greedy best-first) en grafo.

    Expande el nodo con menor h(n), sin tener en cuenta el costo ya
    recorrido. Es rápida pero no garantiza el camino óptimo.

    Parameters
    ----------
    problema : SubproblemaSnake
        Problema a resolver.
    heuristica : callable, optional
        Función h(s) o h(s, objetivo). Por defecto, la del problema o la
        distancia Manhattan.
    guardar_traza : bool
        Si se registra cada iteración.
    """
    t0 = time.perf_counter()
    if heuristica is None:
        heuristica = getattr(problema, "heuristica", heuristica_manhattan)
    nombre_h = getattr(heuristica, "__name__", "manhattan").replace(
        "heuristica_", ""
    )

    s = problema.estado_inicial
    cod = problema.codigo(s)
    v = _evaluar_heuristica(heuristica, problema, s)
    nodo = Nodo(s, None, None, 0.0, cod, heuristica=v, valor_f=v)

    # Frontera ordenada por h(n); un estado se marca al generarlo.
    frontera = ListaPrioritaria()
    frontera.push(nodo, v)
    explorados: Set[str] = {cod}
    celdas_expandidas: Set[Celda] = set()
    traza: List[PasoBusqueda] = []

    nodos_generados = 1
    nodos_expandidos = 0
    max_frontera = 1

    while not frontera.is_empty():
        max_frontera = max(max_frontera, len(frontera))
        nodo = frontera.pop()
        nodos_expandidos += 1
        if hasattr(nodo.estado, "cabeza"):
            celdas_expandidas.add(nodo.estado.cabeza)

        if guardar_traza:
            _capturar_paso(
                traza,
                nodos_expandidos,
                nodo,
                len(frontera),
                nodos_generados,
                nodos_expandidos,
                frontera.elementos(),
                celdas_expandidas,
            )

        if problema.test_objetivo(nodo.estado):
            return _finalizar_resultado(
                problema,
                "Avara",
                nombre_h,
                nodo,
                nodos_generados,
                nodos_expandidos,
                max_frontera,
                t0,
                False,
                traza,
            )

        for a in problema.acciones_aplicables(nodo.estado):
            hijo = nodo_hijo(problema, nodo, a)
            if hijo.codigo not in explorados:
                explorados.add(hijo.codigo)
                v = _evaluar_heuristica(heuristica, problema, hijo.estado)
                hijo.heuristica = v
                hijo.valor_f = v
                nodos_generados += 1
                frontera.push(hijo, v)
                assert len(frontera) < max_size

    return _finalizar_resultado(
        problema,
        "Avara",
        nombre_h,
        None,
        nodos_generados,
        nodos_expandidos,
        max_frontera,
        t0,
        False,
        traza,
    )


def a_estrella(
    problema: Any,
    heuristica: Optional[Callable] = None,
    guardar_traza: bool = False,
) -> Optional[Nodo]:
    """Búsqueda A* en grafo.

    Expande el nodo con menor f(n) = g(n) + h(n). Es completa y óptima
    cuando h es admisible, como la distancia Manhattan o la euclídea.

    Parameters
    ----------
    problema : SubproblemaSnake
        Problema a resolver.
    heuristica : callable, optional
        Función h(s) o h(s, objetivo). Por defecto, la del problema o la
        distancia Manhattan.
    guardar_traza : bool
        Si se registra cada iteración.
    """
    t0 = time.perf_counter()
    if heuristica is None:
        heuristica = getattr(problema, "heuristica", heuristica_manhattan)
    nombre_h = getattr(heuristica, "__name__", "manhattan").replace(
        "heuristica_", ""
    )

    s = problema.estado_inicial
    cod = problema.codigo(s)
    h0 = _evaluar_heuristica(heuristica, problema, s)
    nodo = Nodo(s, None, None, 0.0, cod, heuristica=h0, valor_f=h0)

    frontera = ListaPrioritaria()
    frontera.push(nodo, h0)
    explorados: Dict[str, float] = {cod: 0.0}
    celdas_expandidas: Set[Celda] = set()
    traza: List[PasoBusqueda] = []

    nodos_generados = 1
    nodos_expandidos = 0
    max_frontera = 1

    while not frontera.is_empty():
        max_frontera = max(max_frontera, len(frontera))
        nodo = frontera.pop()

        # Se descartan copias obsoletas con un g(n) peor que el
        # conocido.
        if nodo.costo_camino > explorados.get(nodo.codigo, math.inf) + 1e-9:
            continue

        nodos_expandidos += 1
        if hasattr(nodo.estado, "cabeza"):
            celdas_expandidas.add(nodo.estado.cabeza)

        if guardar_traza:
            _capturar_paso(
                traza,
                nodos_expandidos,
                nodo,
                len(frontera),
                nodos_generados,
                nodos_expandidos,
                frontera.elementos(),
                celdas_expandidas,
            )

        if problema.test_objetivo(nodo.estado):
            # El óptimo solo está garantizado con heurísticas
            # admisibles.
            es_opt = nombre_h in ("manhattan", "euclidea", "heuristica")
            return _finalizar_resultado(
                problema,
                "A*",
                nombre_h,
                nodo,
                nodos_generados,
                nodos_expandidos,
                max_frontera,
                t0,
                es_opt,
                traza,
            )

        for a in problema.acciones_aplicables(nodo.estado):
            hijo = nodo_hijo(problema, nodo, a)
            cod = hijo.codigo
            g = hijo.costo_camino
            # Solo se agrega si el estado es nuevo o el camino es mejor.
            if (cod not in explorados) or (g + 1e-9 < explorados[cod]):
                explorados[cod] = g
                h = _evaluar_heuristica(heuristica, problema, hijo.estado)
                f = g + h
                hijo.heuristica = h
                hijo.valor_f = f
                nodos_generados += 1
                frontera.push(hijo, f)
                assert len(frontera) < max_size

    return _finalizar_resultado(
        problema,
        "A*",
        nombre_h,
        None,
        nodos_generados,
        nodos_expandidos,
        max_frontera,
        t0,
        False,
        traza,
    )


def busqueda_haz(
    problema: Any,
    heuristica: Optional[Callable] = None,
    k: int = 3,
    guardar_traza: bool = False,
) -> Optional[Nodo]:
    """Búsqueda en haz (beam search) con ancho ``k``.

    Funciona como A*, pero en cada iteración conserva solo los ``k``
    mejores nodos de la frontera según f(n) = g(n) + h(n). Usa memoria
    acotada a cambio de perder completitud y optimalidad.

    Parameters
    ----------
    problema : SubproblemaSnake
        Problema a resolver.
    heuristica : callable, optional
        Función h(s) o h(s, objetivo). Por defecto, la del problema o la
        distancia Manhattan.
    k : int
        Ancho del haz.
    guardar_traza : bool
        Si se registra cada iteración.
    """
    t0 = time.perf_counter()
    if heuristica is None:
        heuristica = getattr(problema, "heuristica", heuristica_manhattan)
    nombre_h = getattr(heuristica, "__name__", "manhattan").replace(
        "heuristica_", ""
    )
    nombre_alg = f"Beam(k={k})"

    s = problema.estado_inicial
    cod = problema.codigo(s)
    h0 = _evaluar_heuristica(heuristica, problema, s)
    raiz = Nodo(s, None, None, 0.0, cod, heuristica=h0, valor_f=h0)

    frontera: List[Nodo] = [raiz]
    explorados: Dict[str, float] = {cod: 0.0}
    celdas_expandidas: Set[Celda] = set()
    traza: List[PasoBusqueda] = []

    nodos_generados = 1
    nodos_expandidos = 0
    max_frontera = 1

    while len(frontera) > 0:
        # Se ordena por f(n), desempatando por h(n), y se corta a k
        # nodos.
        frontera.sort(key=lambda n: (n.valor_f, n.heuristica))
        if len(frontera) > k:
            frontera = frontera[:k]

        max_frontera = max(max_frontera, len(frontera))
        nodo = frontera.pop(0)

        nodos_expandidos += 1
        if hasattr(nodo.estado, "cabeza"):
            celdas_expandidas.add(nodo.estado.cabeza)

        if guardar_traza:
            _capturar_paso(
                traza,
                nodos_expandidos,
                nodo,
                len(frontera),
                nodos_generados,
                nodos_expandidos,
                frontera,
                celdas_expandidas,
            )

        if problema.test_objetivo(nodo.estado):
            return _finalizar_resultado(
                problema,
                nombre_alg,
                nombre_h,
                nodo,
                nodos_generados,
                nodos_expandidos,
                max_frontera,
                t0,
                False,
                traza,
            )

        for a in problema.acciones_aplicables(nodo.estado):
            hijo = nodo_hijo(problema, nodo, a)
            cod = hijo.codigo
            g = hijo.costo_camino
            if (cod not in explorados) or (g + 1e-9 < explorados[cod]):
                explorados[cod] = g
                h = _evaluar_heuristica(heuristica, problema, hijo.estado)
                hijo.heuristica = h
                hijo.valor_f = g + h
                nodos_generados += 1
                frontera.append(hijo)

        # Poda inmediata de la frontera a los k mejores nodos.
        if len(frontera) > k:
            frontera.sort(key=lambda n: (n.valor_f, n.heuristica))
            frontera = frontera[:k]

    return _finalizar_resultado(
        problema,
        nombre_alg,
        nombre_h,
        None,
        nodos_generados,
        nodos_expandidos,
        max_frontera,
        t0,
        False,
        traza,
    )


# Nombres alternativos de los algoritmos.
costo_uniforme = dijkstra
A_star = a_estrella
beam_search = busqueda_haz


# =============================================================================
# 5. Ejecución por nombre y comparación sobre una instancia
# =============================================================================


def resolver_subproblema(
    problema: SubproblemaSnake,
    algoritmo: str = "A*",
    heuristica: str = "manhattan",
    k_haz: int = 3,
    guardar_traza: bool = False,
) -> ResultadoBusqueda:
    """Ejecuta un algoritmo elegido por nombre y devuelve su resultado.

    Parameters
    ----------
    problema : SubproblemaSnake
        Problema a resolver.
    algoritmo : str
        ``"BFS"``, ``"DFS"``, ``"UCS"``, ``"Avara"``, ``"A*"`` o
        ``"Beam"``, sin distinguir mayúsculas.
    heuristica : str
        Clave de ``CATALOGO_HEURISTICAS``.
    k_haz : int
        Ancho del haz para la búsqueda en haz.
    guardar_traza : bool
        Si se registra cada iteración.

    Returns
    -------
    ResultadoBusqueda
        Resultado con todas las métricas de la ejecución.

    Raises
    ------
    ValueError
        Si el algoritmo no existe.
    """
    alg = algoritmo.upper().strip()
    fn_h = CATALOGO_HEURISTICAS.get(
        heuristica.lower().strip(), heuristica_manhattan
    )

    if alg in ("BFS", "ANCHURA"):
        anchura(problema, guardar_traza=guardar_traza)
    elif alg in ("DFS", "PROFUNDIDAD"):
        profundidad(problema, guardar_traza=guardar_traza)
    elif alg in ("UCS", "DIJKSTRA", "COSTO_UNIFORME"):
        dijkstra(problema, guardar_traza=guardar_traza)
    elif alg in ("AVARA", "GREEDY"):
        avara(problema, heuristica=fn_h, guardar_traza=guardar_traza)
    elif alg in ("A*", "A_ESTRELLA", "ASTAR"):
        a_estrella(problema, heuristica=fn_h, guardar_traza=guardar_traza)
    elif alg.startswith("BEAM") or alg.startswith("HAZ"):
        busqueda_haz(
            problema, heuristica=fn_h, k=k_haz, guardar_traza=guardar_traza
        )
    else:
        raise ValueError(f"Algoritmo '{algoritmo}' no soportado.")

    return problema.ultimo_resultado


def comparar_algoritmos_en_instancia(
    estado: EstadoSnake,
    lambda_riesgo: float = 0.0,
    k_haz: int = 3,
) -> List[ResultadoBusqueda]:
    """Compara los seis algoritmos sobre un mismo estado.

    Ejecuta cada algoritmo con cada heurística aplicable y marca en
    ``es_optimo`` si su costo coincide con el costo óptimo, tomado de
    UCS y de A* con distancia Manhattan.

    Returns
    -------
    list of ResultadoBusqueda
        Un resultado por configuración.
    """
    configuraciones = [
        ("BFS", "ninguna"),
        ("DFS", "ninguna"),
        ("UCS", "ninguna"),
        ("Avara", "manhattan"),
        ("Avara", "euclidea"),
        ("Avara", "riesgo"),
        ("A*", "manhattan"),
        ("A*", "euclidea"),
        ("A*", "riesgo"),
        ("Beam", "manhattan"),
    ]

    resultados: List[ResultadoBusqueda] = []
    for alg, heur in configuraciones:
        prob = SubproblemaSnake(
            estado_inicial=estado, lambda_riesgo=lambda_riesgo
        )
        res = resolver_subproblema(
            prob,
            algoritmo=alg,
            heuristica=heur,
            k_haz=k_haz,
            guardar_traza=False,
        )
        resultados.append(res)

    # Costo óptimo de referencia, garantizado por UCS y por A* con h_1.
    costos_opt = [
        r.costo_total
        for r in resultados
        if r.exito
        and r.algoritmo in ("UCS", "A*")
        and r.heuristica in ("ninguna", "manhattan")
    ]
    costo_minimo = min(costos_opt) if costos_opt else math.inf

    for r in resultados:
        if r.exito and math.isclose(
            r.costo_total, costo_minimo, rel_tol=1e-6, abs_tol=1e-6
        ):
            r.es_optimo = True
        else:
            r.es_optimo = False

    return resultados
