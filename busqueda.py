"""
Módulo busqueda.py
Estructura basada en los talleres del curso (busqueda.py, busqueda_2.py y utils_busqueda.py).
Incluye:
- ListaPrioritaria (respaldada por heapq con interfaz de los talleres)
- Nodo, nodo_hijo, solucion, depth, camino_codigos, is_cycle
- Algoritmos no informados en grafo: anchura (BFS), profundidad (DFS), backtracking, dijkstra (UCS)
- Algoritmos informados en grafo: avara, a_estrella (A*), busqueda_haz (Beam Search)
- Heurísticas: heuristica_manhattan, heuristica_euclidea, heuristica_riesgo
- Registro de métricas y trazas paso a paso para la animación de la frontera y visitados.
"""

from collections import deque, namedtuple
from dataclasses import dataclass, field
import heapq
import inspect
import itertools
import math
import time
from typing import List, Tuple, Optional, Dict, Callable, Any, Set, Iterable

from ambientes import Celda, EstadoSnake, SubproblemaSnake, calcular_riesgo_celda

# Límites amplios para tableros de Snake (10x10 hasta 20x20)
max_size = 50000
max_size_nodo = 2500

Tupla = namedtuple('Tupla', ['elemento', 'valor'])


# =============================================================================
# 1. ESTRUCTURAS DE DATOS DE LOS TALLERES (ListaPrioritaria y Nodo)
# =============================================================================

class ListaPrioritaria:
    """
    Cola de prioridad compatible con busqueda.py / busqueda_2.py de los talleres,
    optimizada internamente con heapq (como en utils_busqueda.py).
    """

    def __init__(self):
        self._heap: List[Tuple[float, int, Any]] = []
        self._counter = itertools.count()

    def __len__(self) -> int:
        return len(self._heap)

    def push(self, elemento: Any, valor: float) -> None:
        heapq.heappush(self._heap, (float(valor), next(self._counter), elemento))

    def pop(self, with_value: bool = False) -> Any:
        if not self._heap:
            raise IndexError("pop desde una ListaPrioritaria vacía")
        valor, _, elemento = heapq.heappop(self._heap)
        if with_value:
            return Tupla(elemento, valor)
        return elemento

    def remove(self, elemento: Any) -> None:
        for i, (_, _, el) in enumerate(self._heap):
            if el == elemento:
                self._heap.pop(i)
                heapq.heapify(self._heap)
                break

    def elementos(self) -> List[Any]:
        return [el for _, _, el in sorted(self._heap)]

    @property
    def pila(self) -> List[Tupla]:
        return [Tupla(el, val) for val, _, el in sorted(self._heap)]

    def is_empty(self) -> bool:
        return len(self._heap) == 0

    def __str__(self) -> str:
        elementos_ordenados = sorted(self._heap)
        interior = ", ".join(f"({el},{val})" for val, _, el in elementos_ordenados)
        return f"[{interior}]"


class Nodo:
    """
    Clase Nodo de los talleres del curso.
    Mantiene los atributos originales (estado, madre, accion, costo_camino, codigo)
    y añade metadatos para visualización y métricas de la rúbrica.
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
        self.estado = estado
        self.madre = madre
        self.accion = accion
        self.costo_camino = float(costo_camino)
        self.codigo = codigo
        self.heuristica = float(heuristica)
        self.valor_f = float(valor_f) if valor_f is not None else float(costo_camino + heuristica)
        self._profundidad: int = 0 if madre is None else (madre._profundidad + 1)

        # Atributos de telemetría (se llenan en el nodo solución)
        self.nodos_generados: int = 0
        self.nodos_expandidos: int = 0
        self.max_frontera: int = 0
        self.tiempo_ms: float = 0.0
        self.traza: List["PasoBusqueda"] = []


def nodo_hijo(problema: Any, madre: Nodo, accion: str) -> Nodo:
    """
    Función estándar de los talleres para generar un nodo sucesor.
    """
    estado = problema.transicion(madre.estado, accion)
    costo_camino = madre.costo_camino + problema.costo(madre.estado, accion)
    codigo = problema.codigo(estado)
    return Nodo(estado, madre, accion, costo_camino, codigo)


def solucion(n: Optional[Nodo]) -> List[str]:
    """Devuelve la secuencia de acciones desde la raíz hasta el nodo n (iterativo)."""
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
    """Devuelve la secuencia de casillas (x, y) recorridas por la cabeza hasta n."""
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
    """Verifica si el código del nodo ya apareció en su propia rama ancestral."""
    codigo_actual = nodo.codigo
    actual = nodo.madre
    while actual is not None:
        if actual.codigo == codigo_actual:
            return True
        actual = actual.madre
    return False


# =============================================================================
# 2. HEURÍSTICAS (ESTIMADORES DE PÉRDIDA FUTURA h(n))
# =============================================================================

def _obtener_objetivos(estado: EstadoSnake, objetivo: Optional[Celda] = None) -> Tuple[Celda, ...]:
    if objetivo is not None:
        if estado.frutas_extra and objetivo == estado.fruta:
            return estado.todas_las_frutas
        return (objetivo,)
    return estado.todas_las_frutas


def heuristica_manhattan(estado: EstadoSnake, objetivo: Optional[Celda] = None) -> float:
    """
    h_1(s): Distancia Manhattan estándar o toroidal hacia la fruta (u objetivo).
    Admisible y consistente en Snake tanto con costo unitario como con costo de riesgo.
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


def heuristica_euclidea(estado: EstadoSnake, objetivo: Optional[Celda] = None) -> float:
    """
    h_2(s): Distancia Euclídea estándar o toroidal.
    Admisible y dominada por Manhattan: h_2(s) <= h_1(s) <= h*(s).
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


def heuristica_riesgo(estado: EstadoSnake, objetivo: Optional[Celda] = None) -> float:
    """
    h_3(s): Heurística informada por pérdida de espacio local.
    Suma a la distancia Manhattan una penalización por vecinos bloqueados r(h).
    """
    base = heuristica_manhattan(estado, objetivo)
    return base + 2.0 * calcular_riesgo_celda(estado)


CATALOGO_HEURISTICAS: Dict[str, Callable] = {
    "manhattan": heuristica_manhattan,
    "euclidea": heuristica_euclidea,
    "riesgo": heuristica_riesgo,
}


def _evaluar_heuristica(fn_heuristica: Callable, problema: Any, estado: EstadoSnake) -> float:
    """Evalúa la función heurística soportando firmas f(s) o f(s, objetivo)."""
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
# 3. ESTRUCTURAS DE TRAZA Y MÉTRICAS (PARA ANIMACIÓN PASO A PASO Y TABLAS)
# =============================================================================

@dataclass
class PasoBusqueda:
    """Fotografía del estado interno del algoritmo en una iteración."""
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
    """Resumen completo de métricas y traza de una ejecución de búsqueda."""
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
    """Guarda el estado interno de la búsqueda para los controles Play/Pause/Paso."""
    celda_act = nodo.estado.cabeza if hasattr(nodo.estado, "cabeza") else (0, 0)
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
    """Empaqueta las métricas en el Nodo solución y en problema.ultimo_resultado."""
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
# 4. ALGORITMOS DE BÚSQUEDA EN GRAFO (6 ALGORITMOS PRINCIPALES + BACKTRACKING)
# =============================================================================

def anchura(problema: Any, guardar_traza: bool = False) -> Optional[Nodo]:
    """
    1. Método de búsqueda primero en anchura (BFS) en grafo.
    Óptimo cuando el costo por paso es constante (lambda_riesgo == 0).
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

    if problema.test_objetivo(raiz.estado):
        if guardar_traza and hasattr(raiz.estado, "cabeza"):
            celdas_expandidas.add(raiz.estado.cabeza)
            _capturar_paso(traza, 1, raiz, 0, 1, 1, [], celdas_expandidas)
        return _finalizar_resultado(
            problema, "BFS", "ninguna", raiz, 1, 0, 1, t0, True, traza
        )

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
    """
    2. Método de búsqueda primero en profundidad (DFS) en grafo.
    Ciega y no óptima: encuentra caminos largos e irregulares (contraste).
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
    """Método de búsqueda backtracking recursivo de los talleres."""
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
    """
    3. Búsqueda de Costo Uniforme (UCS / Dijkstra) de busqueda_2.py.
    Minimiza la pérdida acumulada g(n) = costo_camino. Es siempre óptima.
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
    """
    4. Búsqueda Avara (Greedy Best-First) de busqueda.py (Taller 10).
    Expande guiándose únicamente por la pérdida futura estimada f(n) = h(n).
    """
    t0 = time.perf_counter()
    if heuristica is None:
        heuristica = getattr(problema, "heuristica", heuristica_manhattan)
    nombre_h = getattr(heuristica, "__name__", "manhattan").replace("heuristica_", "")

    s = problema.estado_inicial
    cod = problema.codigo(s)
    v = _evaluar_heuristica(heuristica, problema, s)
    nodo = Nodo(s, None, None, 0.0, cod, heuristica=v, valor_f=v)

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
    """
    5. Algoritmo A* (A-Estrella) en grafo.
    Minimiza la pérdida total esperada f(n) = g(n) + h(n).
    Es completo y óptimo cuando h(n) es admisible (manhattan o euclidea).
    """
    t0 = time.perf_counter()
    if heuristica is None:
        heuristica = getattr(problema, "heuristica", heuristica_manhattan)
    nombre_h = getattr(heuristica, "__name__", "manhattan").replace("heuristica_", "")

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
    """
    6. Búsqueda en Haz (Beam Search con ancho de haz k).
    Mantiene en memoria únicamente los k mejores nodos según f(n) = g(n) + h(n).
    Permite analizar el compromiso entre memoria acotada y completitud/optimalidad.
    """
    t0 = time.perf_counter()
    if heuristica is None:
        heuristica = getattr(problema, "heuristica", heuristica_manhattan)
    nombre_h = getattr(heuristica, "__name__", "manhattan").replace("heuristica_", "")
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

        # Podamos inmediatamente la frontera a un máximo de k elementos
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


# Alias en inglés/estándar por si algún script los invoca con otro nombre
costo_uniforme = dijkstra
A_star = a_estrella
beam_search = busqueda_haz


# =============================================================================
# 5. DESPACHADOR UNIFICADO Y COMPARADOR SOBRE UNA MISMA INSTANCIA
# =============================================================================

def resolver_subproblema(
    problema: SubproblemaSnake,
    algoritmo: str = "A*",
    heuristica: str = "manhattan",
    k_haz: int = 3,
    guardar_traza: bool = False,
) -> ResultadoBusqueda:
    """
    Ejecuta el algoritmo indicado sobre 'problema' y retorna su ResultadoBusqueda
    con todas las métricas exigidas en la guía oficial.
    """
    alg = algoritmo.upper().strip()
    fn_h = CATALOGO_HEURISTICAS.get(heuristica.lower().strip(), heuristica_manhattan)

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
        busqueda_haz(problema, heuristica=fn_h, k=k_haz, guardar_traza=guardar_traza)
    else:
        raise ValueError(f"Algoritmo '{algoritmo}' no soportado.")

    return problema.ultimo_resultado


def comparar_algoritmos_en_instancia(
    estado: EstadoSnake,
    lambda_riesgo: float = 0.0,
    k_haz: int = 3,
) -> List[ResultadoBusqueda]:
    """
    Ejecuta los 6 algoritmos y las distintas heurísticas sobre el mismo EstadoSnake,
    verificando empíricamente si la solución hallada alcanza el costo óptimo.
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
        prob = SubproblemaSnake(estado_inicial=estado, lambda_riesgo=lambda_riesgo)
        res = resolver_subproblema(
            prob, algoritmo=alg, heuristica=heur, k_haz=k_haz, guardar_traza=False
        )
        resultados.append(res)

    # Costo óptimo de referencia garantizado por UCS y A*(manhattan)
    costos_opt = [
        r.costo_total
        for r in resultados
        if r.exito and r.algoritmo in ("UCS", "A*") and r.heuristica in ("ninguna", "manhattan")
    ]
    costo_minimo = min(costos_opt) if costos_opt else math.inf

    for r in resultados:
        if r.exito and math.isclose(r.costo_total, costo_minimo, rel_tol=1e-6, abs_tol=1e-6):
            r.es_optimo = True
        else:
            r.es_optimo = False

    return resultados