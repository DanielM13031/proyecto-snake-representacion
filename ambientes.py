"""Entorno Snake, su espacio de estados y sus variantes.

El módulo contiene:

1. ``EstadoSnake``: representación inmutable del estado s = (C, f).
2. ``SubproblemaSnake``: espacio de estados para llevar la cabeza hasta
   una fruta o hasta la cola. Lo usan los algoritmos de ``busqueda.py``.
3. ``Snake``: entorno secuencial de la partida completa, con ``reset``,
   ``step`` y ``render``.
4. ``SnakeJuegoAdversario``: juego bipersonal de suma cero para la
   variante en la que un oponente coloca la fruta.
5. ``dibujar_tablero_snake``: dibujo del tablero con matplotlib.
"""

from dataclasses import dataclass
from time import sleep
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

# Una celda del tablero es una pareja de coordenadas enteras (x, y).
Celda = Tuple[int, int]

# Vector de desplazamiento delta(a) asociado a cada acción.
DIRECCIONES: Dict[str, Tuple[int, int]] = {
    "norte": (0, 1),
    "sur": (0, -1),
    "este": (1, 0),
    "oeste": (-1, 0),
}


@dataclass(frozen=True)
class EstadoSnake:
    """Estado s = (C, f) de una partida de Snake.

    Es inmutable, de modo que puede usarse como clave de diccionarios y
    compartirse entre nodos de búsqueda sin copiarlo.

    Attributes
    ----------
    cuerpo : tuple of Celda
        Celdas ordenadas (c_1, ..., c_L); c_1 es la cabeza y c_L la
        cola.
    fruta : Celda or None
        Posición de la fruta principal.
    tamano : int
        Lado N del tablero de N x N celdas.
    toroidal : bool
        Si el tablero no tiene bordes y las coordenadas se envuelven.
    obstaculos : tuple of Celda
        Celdas ocupadas por muros fijos.
    frutas_extra : tuple of Celda
        Frutas adicionales en la variante de varias frutas.
    turno : str
        ``"serpiente"`` o ``"fruta"``; solo se usa en la variante
        adversaria.
    """

    cuerpo: Tuple[Celda, ...]
    fruta: Optional[Celda]
    tamano: int = 10
    toroidal: bool = False
    obstaculos: Tuple[Celda, ...] = ()
    frutas_extra: Tuple[Celda, ...] = ()
    turno: str = "serpiente"

    @property
    def cabeza(self) -> Celda:
        """Celda de la cabeza, c_1."""
        return self.cuerpo[0]

    @property
    def cola(self) -> Celda:
        """Celda de la cola, c_L."""
        return self.cuerpo[-1]

    @property
    def longitud(self) -> int:
        """Longitud L = |C| de la serpiente."""
        return len(self.cuerpo)

    @property
    def todas_las_frutas(self) -> Tuple[Celda, ...]:
        """Tupla con la fruta principal seguida de las frutas extra."""
        if self.fruta is None:
            return self.frutas_extra
        return (self.fruta,) + self.frutas_extra

    def __str__(self) -> str:
        """Resumen legible del estado: cabeza, longitud y fruta."""
        return (
            f"EstadoSnake(cabeza={self.cabeza}, L={self.longitud}, "
            f"fruta={self.fruta})"
        )


# =============================================================================
# Funciones auxiliares de dinámica y riesgo
# =============================================================================


def mover_coordenada(
    celda: Celda, accion: str, tamano: int = 10, toroidal: bool = False
) -> Optional[Celda]:
    """Calcula la celda a la que llega ``celda`` al aplicar ``accion``.

    Parameters
    ----------
    celda : Celda
        Celda de partida.
    accion : str
        Una de las claves de ``DIRECCIONES``.
    tamano : int
        Lado del tablero.
    toroidal : bool
        Si las coordenadas se envuelven al salir del tablero.

    Returns
    -------
    Celda or None
        La nueva celda h' = c + delta(a), o ``None`` si sale del tablero
        en el modo con bordes.

    Raises
    ------
    ValueError
        Si ``accion`` no es una dirección válida.
    """
    if accion not in DIRECCIONES:
        raise ValueError(
            f"Acción inválida: {accion}. Use {list(DIRECCIONES.keys())}"
        )
    dx, dy = DIRECCIONES[accion]
    nx, ny = celda[0] + dx, celda[1] + dy
    if toroidal:
        return (nx % tamano, ny % tamano)
    if 0 <= nx < tamano and 0 <= ny < tamano:
        return (nx, ny)
    return None


def obtener_acciones_aplicables(estado: EstadoSnake) -> List[str]:
    """Devuelve las acciones aplicables en el estado s = (C, f).

    Una acción es aplicable si la nueva cabeza no sale del tablero,
    salvo en el modo toroidal, y no choca con un obstáculo ni con el
    cuerpo. Para L >= 2 se excluye el giro de 180 grados hacia el cuello
    c_2. La celda de la cola c_L sí es válida porque se libera en el
    mismo turno.
    """
    acciones: List[str] = []
    cabeza = estado.cabeza
    cuello = estado.cuerpo[1] if estado.longitud >= 2 else None
    # La cola c_L se libera al avanzar: solo bloquean c_1, ..., c_{L-1}.
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
    """Modelo de transición determinista T(s, a).

    Si la nueva cabeza h' cae sobre una fruta y ``comer_fruta`` es
    verdadero, la serpiente crece una celda. En otro caso el cuerpo
    avanza y la cola libera su celda. Las frutas no se reubican aquí;
    eso lo hace el entorno ``Snake``.

    Raises
    ------
    ValueError
        Si la acción saca la cabeza del tablero.
    """
    nueva_h = mover_coordenada(
        estado.cabeza, accion, estado.tamano, estado.toroidal
    )
    if nueva_h is None:
        raise ValueError(
            f"La acción {accion} sale del tablero desde {estado.cabeza}."
        )

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
    """Calcula el riesgo local r(h') en [0, 1] de la cabeza del estado.

    Cuenta cuántas de las tres direcciones de avance, todas menos la del
    cuello, están bloqueadas por el cuerpo, un obstáculo o el borde, y
    divide entre 3.
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
# 1. Espacio de estados del subproblema
# =============================================================================


class SubproblemaSnake:
    """Subproblema determinista de llevar la cabeza hasta una celda.

    La celda objetivo es la fruta actual o, en el chequeo de seguridad,
    la cola. Implementa los métodos que usan los algoritmos de
    ``busqueda.py``: ``estado_inicial``, ``acciones_aplicables``,
    ``transicion``, ``test_objetivo``, ``costo`` y ``codigo``.

    Parameters
    ----------
    estado_inicial : EstadoSnake, optional
        Estado de partida. Por defecto, una serpiente de longitud 1 en
        (5, 5) con la fruta en (8, 8).
    objetivo : Celda, optional
        Celda a alcanzar. Por defecto, la fruta del estado inicial.
    lambda_riesgo : float
        Peso lambda del riesgo en el costo por paso.
    comer_en_objetivo : bool
        Si la serpiente crece al llegar al objetivo.
    codigo_cuerpo_completo : bool
        Si el código del estado incluye todo el cuerpo y no solo la
        cabeza.
    """

    def __init__(
        self,
        estado_inicial: Optional[EstadoSnake] = None,
        objetivo: Optional[Celda] = None,
        lambda_riesgo: float = 0.0,
        comer_en_objetivo: bool = True,
        codigo_cuerpo_completo: bool = False,
    ):
        """Crea el subproblema con los parámetros indicados."""
        if estado_inicial is None:
            estado_inicial = EstadoSnake(cuerpo=((5, 5),), fruta=(8, 8))
        self.estado_inicial = estado_inicial
        self.objetivo = (
            objetivo if objetivo is not None else estado_inicial.fruta
        )
        self.lambda_riesgo = lambda_riesgo
        self.comer_en_objetivo = comer_en_objetivo
        self.codigo_cuerpo_completo = codigo_cuerpo_completo

    def acciones_aplicables(self, estado: EstadoSnake) -> List[str]:
        """Devuelve la lista de acciones aplicables desde ``estado``."""
        return obtener_acciones_aplicables(estado)

    def transicion(self, estado: EstadoSnake, accion: str) -> EstadoSnake:
        """Devuelve el estado que resulta de aplicar ``accion``.

        La serpiente solo crece si la acción la lleva al objetivo y
        ``comer_en_objetivo`` es verdadero.
        """
        nueva_h = mover_coordenada(
            estado.cabeza, accion, estado.tamano, estado.toroidal
        )
        es_meta = (self.objetivo is not None) and (nueva_h == self.objetivo)
        return aplicar_transicion(
            estado, accion, comer_fruta=(self.comer_en_objetivo and es_meta)
        )

    def test_objetivo(self, estado: EstadoSnake) -> bool:
        """Indica si la cabeza llegó al objetivo del subproblema."""
        # Con varias frutas, cualquiera de ellas cuenta como objetivo.
        if estado.frutas_extra and self.objetivo == estado.fruta:
            return estado.cabeza in estado.todas_las_frutas
        return self.objetivo is not None and estado.cabeza == self.objetivo

    def costo(self, estado: EstadoSnake, accion: str) -> float:
        """Costo de un paso, c(s, a, s') = 1 + lambda * r(h').

        Con lambda = 0 el costo es 1 y no hace falta simular la
        transición.
        """
        if self.lambda_riesgo <= 0.0:
            return 1.0
        estado_sig = self.transicion(estado, accion)
        return 1.0 + self.lambda_riesgo * calcular_riesgo_celda(estado_sig)

    def codigo(self, estado: EstadoSnake) -> str:
        """Código del estado para la tabla de explorados.

        Por defecto codifica solo la cabeza como ``"x-y"``, lo que acota
        el grafo a las N^2 celdas del tablero. Con
        ``codigo_cuerpo_completo`` codifica todo el cuerpo.
        """
        if self.codigo_cuerpo_completo:
            return "|".join(f"{x}-{y}" for x, y in estado.cuerpo)
        x, y = estado.cabeza
        return f"{x}-{y}"

    def heuristica(self, estado: EstadoSnake) -> float:
        """Heurística por defecto h_1(s): distancia Manhattan a la meta.

        En el modo toroidal usa la distancia Manhattan toroidal y, con
        varias frutas, la distancia a la más cercana. Permite llamar a
        ``avara(problema, problema.heuristica)``.
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
        """Anima una secuencia de acciones desde el estado inicial.

        Parameters
        ----------
        camino : list of str
            Acciones a aplicar en orden.
        sleep_time : float
            Pausa en segundos entre cuadros.
        """
        estados = [self.estado_inicial]
        for accion in camino:
            estados.append(self.transicion(estados[-1], accion))
        for est in estados:
            clear_output(wait=True)
            self.render(est)
            plt.show()
            sleep(sleep_time)


# =============================================================================
# 2. Entorno secuencial de la partida completa
# =============================================================================


class Snake:
    """Entorno por turnos de una partida completa de Snake.

    Ofrece la interfaz que usan ``Episode`` y el benchmark de
    ``utils.py``:

    - ``reset()`` devuelve el estado inicial.
    - ``step(accion)`` devuelve ``(nuevo_estado, recompensa, done)``.
    - ``test_objetivo()`` indica si el tablero quedó lleno.
    - ``render()`` dibuja el tablero con matplotlib.

    Parameters
    ----------
    tamano : int
        Lado N del tablero.
    semilla : int
        Semilla del generador de obstáculos y frutas.
    pos_inicial : Celda, optional
        Posición inicial de la cabeza. Por defecto, el centro.
    fruta_inicial : Celda, optional
        Posición de la primera fruta. Por defecto, al azar.
    toroidal : bool
        Si el tablero no tiene bordes.
    num_obstaculos : int
        Número de muros fijos generados al azar.
    num_frutas : int
        Número de frutas simultáneas en el tablero.
    lambda_riesgo : float
        Peso del riesgo en la pérdida por paso.
    k_max : int
        Máximo de pasos seguidos sin comer antes de terminar.
    ganancia_fruta, ganancia_victoria, perdida_choque : float
        Términos de la función de recompensa.
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
        """Crea el entorno y lo deja listo con ``reset``."""
        self.tamano = tamano
        self.semilla = semilla
        self.pos_inicial_config = pos_inicial
        self.fruta_inicial_config = fruta_inicial
        self.toroidal = toroidal
        self.num_obstaculos = num_obstaculos
        self.num_frutas = max(1, num_frutas)
        self.lambda_riesgo = lambda_riesgo
        self.k_max = k_max

        # Términos de la función de recompensa R.
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
        """Reinicia la partida y devuelve el estado inicial.

        Usar la misma semilla garantiza que distintos agentes enfrenten
        exactamente los mismos obstáculos y la misma secuencia de
        frutas.

        Parameters
        ----------
        semilla : int, optional
            Nueva semilla. Si se omite, se reutiliza la actual.
        """
        if semilla is not None:
            self.semilla = semilla
        self.rng = np.random.default_rng(self.semilla)

        if self.pos_inicial_config is not None:
            cabeza_ini = self.pos_inicial_config
        else:
            cabeza_ini = (self.tamano // 2, self.tamano // 2)

        cuerpo_ini = (cabeza_ini,)

        # Obstáculos fijos, generados de forma reproducible con la
        # semilla.
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

        # Frutas iniciales en celdas libres.
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
        frutas_extra = (
            tuple(frutas_generadas[1:]) if len(frutas_generadas) > 1 else ()
        )

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
        """Elige al azar, de forma uniforme, una celda libre.

        Devuelve ``None`` si no quedan celdas libres.
        """
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

    def acciones_aplicables(
        self, estado: Optional[EstadoSnake] = None
    ) -> List[str]:
        """Devuelve las acciones aplicables en ``estado`` o actual."""
        est = estado if estado is not None else self.estado
        if est is None:
            return []
        return obtener_acciones_aplicables(est)

    def test_objetivo(self, estado: Optional[EstadoSnake] = None) -> bool:
        """Indica si se ganó, es decir, si el tablero está lleno."""
        est = estado if estado is not None else self.estado
        if est is None:
            return False
        meta = (est.tamano * est.tamano) - len(est.obstaculos)
        return est.longitud >= meta

    def step(self, accion: str) -> Tuple[EstadoSnake, float, bool]:
        """Ejecuta una acción y avanza un turno.

        Una acción no aplicable se trata como colisión y termina la
        partida. Si la serpiente come, se reubica la fruta; después se
        revisan los estados terminales.

        Returns
        -------
        tuple
            ``(nuevo_estado, recompensa, done)``.
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

        # Transición del cuerpo y pérdida del paso, 1 + lambda * r(h').
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
            frutas_restantes = [
                f for f in frutas_previas if f != est_trans.cabeza
            ]

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
                    tuple(frutas_restantes[1:])
                    if len(frutas_restantes) > 1
                    else ()
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

        # Estados terminales: k_max pasos sin comer o sin salida.
        if not self.done:
            if self.pasos_sin_comer >= self.k_max:
                self.done = True
                self.causa_fin = (
                    f"Límite de pasos sin comer alcanzado (k_max={self.k_max})"
                )
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
        """Dibuja el estado actual u otro estado, con capas de búsqueda.

        Si no se da un título, se muestra un resumen de la partida.
        Los demás parámetros se pasan a ``dibujar_tablero_snake``.
        """
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
# 3. Juego adversario
# =============================================================================


class SnakeJuegoAdversario:
    """Variante de fruta adversaria como juego de suma cero.

    - El jugador MAX, ``"serpiente"``, mueve la cabeza para sobrevivir y
      comer.
    - El jugador MIN, ``"fruta"``, coloca cada fruta nueva en la celda
      libre que minimiza la utilidad de la serpiente.

    Implementa la interfaz de juego que usa ``AgenteFrutaAdversaria``:
    ``player``, ``acciones``, ``resultado``, ``es_terminal``,
    ``utilidad`` y ``render``.
    """

    def __init__(self, tamano: int = 10, obstaculos: Tuple[Celda, ...] = ()):
        """Crea el juego con la serpiente en el centro del tablero."""
        mitad = tamano // 2
        self.tamano = tamano
        self.estado_inicial = EstadoSnake(
            cuerpo=((mitad, mitad),),
            fruta=(mitad + 2, mitad),
            tamano=tamano,
            obstaculos=obstaculos,
            turno="serpiente",
        )

    def player(self, estado: EstadoSnake) -> str:
        """Devuelve quién juega: ``"serpiente"`` (MAX) o ``"fruta"``."""
        return estado.turno

    def acciones(self, estado: EstadoSnake) -> List:
        """Devuelve las jugadas posibles del jugador en turno.

        Para la serpiente son las direcciones aplicables; para la fruta,
        las celdas libres donde puede aparecer.
        """
        if estado.turno == "serpiente":
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
        """Aplica la jugada del jugador y da el estado nuevo."""
        if estado.turno == "serpiente":
            est_sig = aplicar_transicion(estado, accion, comer_fruta=True)
            comio = (estado.fruta is not None) and (
                est_sig.cabeza == estado.fruta
            )
            if comio:
                # Si comió, el turno pasa a la fruta para colocar la
                # nueva.
                return EstadoSnake(
                    cuerpo=est_sig.cuerpo,
                    fruta=None,
                    tamano=estado.tamano,
                    toroidal=estado.toroidal,
                    obstaculos=estado.obstaculos,
                    turno="fruta",
                )
            return EstadoSnake(
                cuerpo=est_sig.cuerpo,
                fruta=estado.fruta,
                tamano=estado.tamano,
                toroidal=estado.toroidal,
                obstaculos=estado.obstaculos,
                turno="serpiente",
            )
        else:
            # Jugada de la fruta: accion es la celda (x, y) donde
            # aparece.
            return EstadoSnake(
                cuerpo=estado.cuerpo,
                fruta=accion,
                tamano=estado.tamano,
                toroidal=estado.toroidal,
                obstaculos=estado.obstaculos,
                turno="serpiente",
            )

    def es_terminal(self, estado: EstadoSnake) -> bool:
        """Indica si el juego terminó.

        Termina cuando el tablero se llena o cuando el jugador en turno
        no tiene jugadas.
        """
        meta = (estado.tamano * estado.tamano) - len(estado.obstaculos)
        if estado.longitud >= meta:
            return True
        if (
            estado.turno == "serpiente"
            and len(obtener_acciones_aplicables(estado)) == 0
        ):
            return True
        if estado.turno == "fruta" and len(self.acciones(estado)) == 0:
            return True
        return False

    def utilidad(
        self, estado: EstadoSnake, jugador: str = "serpiente"
    ) -> Optional[float]:
        """Devuelve +1000 si gana la serpiente y -1000 si se atrapa."""
        if not self.es_terminal(estado):
            return None
        meta = (estado.tamano * estado.tamano) - len(estado.obstaculos)
        valor = 1000.0 if estado.longitud >= meta else -1000.0
        return valor if jugador == "serpiente" else -valor

    def render(self, estado: EstadoSnake):
        """Dibuja el estado indicando de quién es el turno."""
        dibujar_tablero_snake(
            estado, titulo=f"Modo Adversario | Turno: {estado.turno}"
        )


# =============================================================================
# 4. Dibujo del tablero
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
    """Dibuja el tablero de Snake con matplotlib.

    Las capas se distinguen por color: celdas expandidas en azul claro,
    frontera en amarillo, nodo actual en naranja, camino planeado en
    verde claro, obstáculos en negro, frutas en rojo y el cuerpo con un
    degradado de la cabeza c_1 a la cola c_L.

    Parameters
    ----------
    estado : EstadoSnake
        Estado a dibujar.
    camino, frontera, expandidos : list of Celda, optional
        Capas de la búsqueda.
    celda_actual : Celda, optional
        Celda que el algoritmo está expandiendo.
    titulo : str, optional
        Título del gráfico.
    ax : matplotlib.axes.Axes, optional
        Ejes donde dibujar. Si se omite, se crea una figura y se
        muestra.

    Returns
    -------
    matplotlib.axes.Axes
        Los ejes con el dibujo.
    """
    mostrar_plt = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        ax.clear()

    n = estado.tamano
    step = 1.0 / n

    # Fondo del tablero.
    ax.add_patch(
        patches.Rectangle(
            (0, 0),
            1.0,
            1.0,
            facecolor="cornsilk",
            edgecolor="black",
            linewidth=2,
        )
    )

    # 1. Celdas expandidas durante la búsqueda.
    if expandidos:
        for x, y in expandidos:
            ax.add_patch(
                patches.Rectangle(
                    (x * step, y * step),
                    step,
                    step,
                    facecolor="#b3cde3",
                    alpha=0.65,
                )
            )

    # 2. Celdas en la frontera.
    if frontera:
        for x, y in frontera:
            ax.add_patch(
                patches.Rectangle(
                    (x * step, y * step),
                    step,
                    step,
                    facecolor="#fed976",
                    alpha=0.85,
                )
            )

    # 3. Camino planeado hacia la meta.
    if camino:
        for x, y in camino:
            ax.add_patch(
                patches.Rectangle(
                    (x * step + step * 0.18, y * step + step * 0.18),
                    step * 0.64,
                    step * 0.64,
                    facecolor="#78c679",
                    alpha=0.75,
                )
            )

    # 4. Celda que el algoritmo está expandiendo.
    if celda_actual is not None:
        cx, cy = celda_actual
        ax.add_patch(
            patches.Rectangle(
                (cx * step, cy * step),
                step,
                step,
                facecolor="#fd8d3c",
                edgecolor="red",
                linewidth=2,
                alpha=0.9,
            )
        )

    # 5. Obstáculos fijos.
    for ox, oy in estado.obstaculos:
        ax.add_patch(
            patches.Rectangle(
                (ox * step, oy * step),
                step,
                step,
                facecolor="#252525",
            )
        )

    # 6. Cuerpo, de verde oscuro en la cabeza a verde claro en la cola.
    longitud = estado.longitud
    for idx, (sx, sy) in enumerate(estado.cuerpo):
        if idx == 0:
            color = "#005a32"  # Cabeza c_1.
        elif idx == longitud - 1:
            color = "#a1d99b"  # Cola c_L.
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
                edgecolor="black",
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

    # 7. Frutas.
    for fx, fy in estado.todas_las_frutas:
        circulo = patches.Circle(
            ((fx + 0.5) * step, (fy + 0.5) * step),
            radius=step * 0.36,
            facecolor="#e31a1c",
            edgecolor="darkred",
            linewidth=1.5,
        )
        ax.add_patch(circulo)

    # Líneas de la cuadrícula.
    for j in range(1, n):
        loc = j * step
        ax.add_patch(
            patches.Rectangle(
                (0, loc), 1.0, 0.003, facecolor="gray", alpha=0.4
            )
        )
        ax.add_patch(
            patches.Rectangle(
                (loc, 0), 0.003, 1.0, facecolor="gray", alpha=0.4
            )
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    if titulo:
        # La causa de fin va en su propia línea para que el título
        # quepa.
        if " | Fin:" in titulo and "\nFin:" not in titulo:
            titulo = titulo.replace(" | Fin:", "\nFin:")
        ax.set_title(
            titulo,
            fontsize=9.5,
            weight="bold",
            pad=10,
            linespacing=1.35,
            wrap=True,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="#f8f9fa",
                edgecolor="#cccccc",
                alpha=0.9,
            ),
        )

    if mostrar_plt:
        plt.show()
    return ax
