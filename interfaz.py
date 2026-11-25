"""
Módulo interfaz.py
SnakeLab Studio: aplicación gráfica que se abre como una página aparte.

Arquitectura:
- Este módulo levanta un servidor HTTP pequeño (solo librería estándar) dentro
  del kernel del notebook. Sirve la página de la carpeta ``web/`` y expone una
  API JSON que llama directamente a ambientes.py, busqueda.py y agentes.py.
- La página (web/index.html, web/app.js, web/estilos.css) dibuja el tablero en
  un canvas y anima en el navegador las trazas que devuelve la API.

Uso desde el notebook:
    from interfaz import lanzar_aplicacion_snake
    lanzar_aplicacion_snake()        # en Colab abre un enlace a otra pestaña

Endpoints:
    POST /api/instancia   genera una instancia aleatoria (tamaño, longitud, muros, semilla)
    POST /api/buscar      corre un algoritmo sobre una instancia y devuelve la traza paso a paso
    POST /api/comparar    corre todos los algoritmos sobre la misma instancia
    POST /api/partida     simula una partida completa con AgenteSnake
    POST /api/jugar/*     modo manual: humano contra la IA, paso a paso
    GET  /api/benchmark   resumen del CSV de 100 semillas (si existe)
"""

import json
import math
import socket
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ambientes import (
    Celda,
    EstadoSnake,
    SubproblemaSnake,
    Snake,
    obtener_acciones_aplicables,
    mover_coordenada,
)
from busqueda import resolver_subproblema, comparar_algoritmos_en_instancia
from agentes import AgenteSnake

CARPETA_MODULO = Path(__file__).resolve().parent
CARPETA_WEB = CARPETA_MODULO / "web"
RUTA_CSV = CARPETA_MODULO / "resultados_benchmark_100_semillas.csv"

# Mismas configuraciones que comparar_algoritmos_en_instancia (busqueda.py)
CONFIGS_COMPARACION = [
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

MODOS_DECISION = {
    "ruta_fruta": 0,
    "evasion_cola (ruta insegura)": 1,
    "evasion_cola (sin camino a fruta)": 2,
}


# =============================================================================
# CONVERSIÓN ENTRE JSON Y EstadoSnake
# =============================================================================

def _limpiar(obj: Any) -> Any:
    """Convierte tuplas, tipos de numpy e infinitos a algo que JSON.parse acepte."""
    if isinstance(obj, dict):
        return {str(k): _limpiar(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_limpiar(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        valor = float(obj)
        return None if (math.isinf(valor) or math.isnan(valor)) else round(valor, 4)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _celda(c) -> Celda:
    return (int(c[0]), int(c[1]))


def _estado_desde_json(d: Dict[str, Any]) -> EstadoSnake:
    return EstadoSnake(
        cuerpo=tuple(_celda(c) for c in d["cuerpo"]),
        fruta=_celda(d["fruta"]) if d.get("fruta") is not None else None,
        tamano=int(d.get("N", 10)),
        toroidal=bool(d.get("toroidal", False)),
        obstaculos=tuple(_celda(c) for c in d.get("muros", [])),
    )


def _estado_a_json(e: EstadoSnake) -> Dict[str, Any]:
    return {
        "N": e.tamano,
        "cuerpo": [list(c) for c in e.cuerpo],
        "fruta": list(e.fruta) if e.fruta is not None else None,
        "frutas": [list(f) for f in e.todas_las_frutas],
        "muros": [list(c) for c in e.obstaculos],
        "toroidal": e.toroidal,
    }


def _vecinos(c: Celda, n: int, toroidal: bool) -> List[Celda]:
    res = []
    for acc in ("norte", "sur", "este", "oeste"):
        v = mover_coordenada(c, acc, n, toroidal)
        if v is not None:
            res.append(v)
    return res


def _instancia_aleatoria(
    n: int, longitud: int, n_muros: int, semilla: int, toroidal: bool
) -> EstadoSnake:
    """
    Genera una instancia válida: cuerpo como camino autoevitante desde la cabeza,
    muros en celdas libres (no pegados a la cabeza) y una fruta en una celda libre.
    """
    rng = np.random.default_rng(semilla)
    longitud = max(1, min(longitud, n * n // 2))
    for _ in range(400):
        cabeza = (int(rng.integers(0, n)), int(rng.integers(0, n)))
        cuerpo = [cabeza]
        usadas = {cabeza}
        while len(cuerpo) < longitud:
            opciones = [v for v in _vecinos(cuerpo[-1], n, toroidal) if v not in usadas]
            if not opciones:
                break
            sig = opciones[int(rng.integers(0, len(opciones)))]
            cuerpo.append(sig)
            usadas.add(sig)
        if len(cuerpo) < longitud:
            continue

        cerca_cabeza = set(_vecinos(cabeza, n, toroidal))
        libres = [
            (x, y)
            for x in range(n)
            for y in range(n)
            if (x, y) not in usadas and (x, y) not in cerca_cabeza
        ]
        if len(libres) < n_muros + 1:
            continue
        idx = rng.permutation(len(libres))
        muros = [libres[int(i)] for i in idx[:n_muros]]
        resto = [libres[int(i)] for i in idx[n_muros:]]
        fruta = resto[int(rng.integers(0, len(resto)))]

        estado = EstadoSnake(
            cuerpo=tuple(cuerpo),
            fruta=fruta,
            tamano=n,
            toroidal=toroidal,
            obstaculos=tuple(muros),
        )
        if obtener_acciones_aplicables(estado):
            return estado
    raise ValueError("No se pudo generar una instancia válida con esos parámetros.")


# =============================================================================
# SERVICIOS DE LA API
# =============================================================================

def api_instancia(d: Dict[str, Any]) -> Dict[str, Any]:
    estado = _instancia_aleatoria(
        n=int(d.get("N", 10)),
        longitud=int(d.get("longitud", 1)),
        n_muros=int(d.get("muros", 0)),
        semilla=int(d.get("semilla", 0)),
        toroidal=bool(d.get("toroidal", False)),
    )
    return {"estado": _estado_a_json(estado)}


def api_buscar(d: Dict[str, Any]) -> Dict[str, Any]:
    """Corre un algoritmo con traza y devuelve la animación completa."""
    estado = _estado_desde_json(d["estado"])
    lam = float(d.get("lambda", 0.0))
    prob = SubproblemaSnake(estado_inicial=estado, lambda_riesgo=lam)
    res = resolver_subproblema(
        prob,
        algoritmo=d.get("algoritmo", "A*"),
        heuristica=d.get("heuristica", "manhattan"),
        k_haz=int(d.get("k", 3)),
        guardar_traza=True,
    )

    pasos = [
        {
            "it": p.iteracion,
            "cur": p.celda_actual,
            "acc": p.accion_tomada,
            "g": p.g,
            "h": p.h,
            "f": p.f,
            "prof": p.profundidad,
            "tf": p.tamano_frontera,
            "gen": p.nodos_generados,
            "exp": p.nodos_expandidos,
            "fr": p.frontera_celdas,
            "ex": p.expandidos_celdas,
            "cam": p.camino_parcial,
        }
        for p in res.traza
    ]

    # Secuencia de cuerpos al ejecutar la solución (animación del camino final)
    cuerpos = [list(estado.cuerpo)]
    seguro = None
    if res.exito:
        actual = estado
        for a in res.acciones:
            actual = prob.transicion(actual, a)
            cuerpos.append(list(actual.cuerpo))
        seguro = AgenteSnake._tiene_salida_a_cola(res.estado_final)[0]

    return _limpiar(
        {
            "pasos": pasos,
            "cuerpos": cuerpos,
            "resultado": {
                "algoritmo": res.algoritmo,
                "heuristica": res.heuristica,
                "exito": res.exito,
                "acciones": res.acciones,
                "camino": res.camino_celdas,
                "costo": res.costo_total,
                "longitud": res.longitud_solucion,
                "generados": res.nodos_generados,
                "expandidos": res.nodos_expandidos,
                "max_frontera": res.max_frontera,
                "tiempo_ms": res.tiempo_ms,
                "optimo_teorico": res.es_optimo,
                "seguro": seguro,
            },
        }
    )


def api_comparar(d: Dict[str, Any]) -> Dict[str, Any]:
    """Ejecuta los 10 algoritmos/heurísticas sobre la misma instancia."""
    estado = _estado_desde_json(d["estado"])
    lam = float(d.get("lambda", 0.0))
    k = int(d.get("k", 3))

    # Métricas limpias (sin el costo de guardar la traza)
    resultados = comparar_algoritmos_en_instancia(estado, lambda_riesgo=lam, k_haz=k)

    filas = []
    for (alg, heur), res in zip(CONFIGS_COMPARACION, resultados):
        # Segunda corrida solo para dibujar qué celdas exploró cada algoritmo
        prob = SubproblemaSnake(estado_inicial=estado, lambda_riesgo=lam)
        res_traza = resolver_subproblema(prob, alg, heur, k_haz=k, guardar_traza=True)
        ultimo = res_traza.traza[-1] if res_traza.traza else None
        filas.append(
            {
                "algoritmo": alg,
                "heuristica": heur,
                "nombre": res.algoritmo,
                "exito": res.exito,
                "optimo": res.es_optimo,
                "longitud": res.longitud_solucion,
                "costo": res.costo_total,
                "generados": res.nodos_generados,
                "expandidos": res.nodos_expandidos,
                "max_frontera": res.max_frontera,
                "tiempo_ms": res.tiempo_ms,
                "camino": res.camino_celdas,
                "explorados": ultimo.expandidos_celdas if ultimo else [],
                "frontera": ultimo.frontera_celdas if ultimo else [],
            }
        )
    return _limpiar({"filas": filas})


def _plan_a_celdas(inicio: Celda, acciones: List[str], n: int, toroidal: bool) -> List[Celda]:
    celdas, actual = [], inicio
    for a in acciones:
        actual = mover_coordenada(actual, a, n, toroidal)
        if actual is None:
            break
        celdas.append(actual)
    return celdas


def api_partida(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simula una partida completa con AgenteSnake.
    Para no mandar el cuerpo completo en cada turno, se envía la cabeza y si comió:
    el navegador reconstruye el cuerpo con la misma regla de transición T.
    """
    n = int(d.get("N", 10))
    semilla = int(d.get("semilla", 0))
    toroidal = bool(d.get("toroidal", False))
    lam = float(d.get("lambda", 0.0))
    max_pasos = int(d.get("max_pasos", 2500))

    env = Snake(
        tamano=n,
        semilla=semilla,
        toroidal=toroidal,
        num_obstaculos=int(d.get("muros", 0)),
        num_frutas=int(d.get("frutas", 1)),
        lambda_riesgo=lam,
        k_max=int(d.get("k_max", 200)),
    )
    agente = AgenteSnake(
        algoritmo=d.get("algoritmo", "A*"),
        heuristica=d.get("heuristica", "manhattan"),
        chequeo_seguridad=bool(d.get("seguridad", True)),
        lambda_riesgo=lam,
        k_haz=int(d.get("k", 3)),
    )
    estado = env.reset(semilla=semilla)
    agente.restart()
    agente.states.append(estado)
    inicial = _estado_a_json(estado)

    turnos: List[Dict[str, Any]] = []
    modo_actual = 0
    G = 0.0
    while not env.done and env.pasos < max_pasos:
        n_dec = len(agente.historial_decisiones)
        accion = agente.make_decision()
        decidio = len(agente.historial_decisiones) > n_dec
        if decidio:
            modo_actual = MODOS_DECISION.get(agente.historial_decisiones[-1]["modo"], 0)
        plan = _plan_a_celdas(estado.cabeza, [accion] + list(agente.plan), n, toroidal)
        L_antes = estado.longitud

        estado, recompensa, _ = env.step(accion)
        agente.states.append(estado)
        G += recompensa
        turnos.append(
            {
                "a": accion,
                "h": estado.cabeza,
                "come": estado.longitud > L_antes,
                "fr": estado.todas_las_frutas,
                "m": modo_actual,
                "d": decidio,
                "p": plan,
                "G": G,
            }
        )

    hist = agente.historial_decisiones
    resumen = {
        "causa_fin": env.causa_fin if env.done else f"Corte en {max_pasos} pasos",
        "victoria": env.test_objetivo(env.estado),
        "longitud": env.estado.longitud,
        "frutas": env.frutas_comidas,
        "pasos": env.pasos,
        "ganancia": env.ganancia_total,
        "perdida": env.perdida_total,
        "G": env.ganancia_total - env.perdida_total,
        "decisiones": len(hist),
        "evasiones": agente.pasos_siguiendo_cola,
        "gen_prom": float(np.mean([h["nodos_generados"] for h in hist])) if hist else 0.0,
        "exp_prom": float(np.mean([h["nodos_expandidos"] for h in hist])) if hist else 0.0,
        "tiempo_prom_ms": float(np.mean([h["tiempo_ms"] for h in hist])) if hist else 0.0,
        "meta": n * n - len(estado.obstaculos),
    }
    return _limpiar({"inicial": inicial, "turnos": turnos, "resumen": resumen})


# ---------- Modo manual (humano contra la IA) ----------

_SESIONES: Dict[str, Dict[str, Any]] = {}
_CANDADO = threading.Lock()


def _foto_juego(env: Snake, G: float) -> Dict[str, Any]:
    return {
        "estado": _estado_a_json(env.estado),
        "longitud": env.estado.longitud,
        "frutas": env.frutas_comidas,
        "pasos": env.pasos,
        "G": G,
        "fin": env.done,
        "causa": env.causa_fin,
        "victoria": env.test_objetivo(env.estado),
    }


def _respuesta_juego(sid: str) -> Dict[str, Any]:
    s = _SESIONES[sid]
    return _limpiar(
        {
            "id": sid,
            "humano": _foto_juego(s["env_h"], s["G_h"]),
            "ia": _foto_juego(s["env_ia"], s["G_ia"]),
            "ia_modo": s["ia_modo"],
        }
    )


def api_jugar_nuevo(d: Dict[str, Any]) -> Dict[str, Any]:
    n = int(d.get("N", 10))
    semilla = int(d.get("semilla", 0))
    muros = int(d.get("muros", 0))
    env_h = Snake(tamano=n, semilla=semilla, num_obstaculos=muros, k_max=200)
    env_ia = Snake(tamano=n, semilla=semilla, num_obstaculos=muros, k_max=200)
    agente = AgenteSnake(
        algoritmo=d.get("algoritmo", "A*"),
        heuristica="manhattan",
        chequeo_seguridad=True,
    )
    agente.restart()
    agente.states.append(env_ia.reset(semilla=semilla))
    env_h.reset(semilla=semilla)
    sid = uuid.uuid4().hex[:10]
    with _CANDADO:
        if len(_SESIONES) > 50:
            _SESIONES.clear()
        _SESIONES[sid] = {
            "env_h": env_h,
            "env_ia": env_ia,
            "agente": agente,
            "G_h": 0.0,
            "G_ia": 0.0,
            "ia_modo": 0,
        }
    return _respuesta_juego(sid)


def api_jugar_paso(d: Dict[str, Any]) -> Dict[str, Any]:
    """Un turno: el humano mueve y la IA responde con un movimiento."""
    sid = d["id"]
    with _CANDADO:
        s = _SESIONES[sid]
        if not s["env_h"].done:
            _, r, _ = s["env_h"].step(d["dir"])
            s["G_h"] += r
            env_ia, agente = s["env_ia"], s["agente"]
            # la IA responde solo si el movimiento del humano fue válido
            if not env_ia.done and not s["env_h"].causa_fin.startswith("Colisión"):
                n_dec = len(agente.historial_decisiones)
                accion = agente.make_decision()
                if len(agente.historial_decisiones) > n_dec:
                    s["ia_modo"] = MODOS_DECISION.get(agente.historial_decisiones[-1]["modo"], 0)
                est, r_ia, _ = env_ia.step(accion)
                agente.states.append(est)
                s["G_ia"] += r_ia
        return _respuesta_juego(sid)


def api_jugar_pista(d: Dict[str, Any]) -> Dict[str, Any]:
    """Camino que A* sugiere desde el estado actual del humano."""
    s = _SESIONES[d["id"]]
    estado = s["env_h"].estado
    if s["env_h"].done or estado.fruta is None:
        return {"camino": [], "seguro": None}
    prob = SubproblemaSnake(estado_inicial=estado, objetivo=estado.fruta)
    res = resolver_subproblema(prob, algoritmo="A*", heuristica="manhattan")
    if not res.exito:
        return {"camino": [], "seguro": False}
    seguro = AgenteSnake._tiene_salida_a_cola(res.estado_final)[0]
    return _limpiar({"camino": res.camino_celdas, "seguro": seguro})


def api_benchmark(_: Dict[str, Any]) -> Dict[str, Any]:
    """Resumen del benchmark de 100 semillas, agrupado por configuración real (incluye λ)."""
    if not RUTA_CSV.exists():
        return {"disponible": False}
    import pandas as pd

    df = pd.read_csv(RUTA_CSV)
    df["fin_kmax"] = df["causa_fin"].astype(str).str.startswith("Límite")
    df["corte"] = df["causa_fin"].astype(str).eq("En curso")
    claves = ["chequeo_seguridad", "algoritmo", "heuristica", "lambda_riesgo"]
    g = df.groupby(claves, sort=False)
    res = g.agg(
        longitud=("longitud_final", "mean"),
        longitud_std=("longitud_final", "std"),
        victorias=("victoria", "mean"),
        fin_kmax=("fin_kmax", "mean"),
        corte=("corte", "mean"),
        pasos_fruta=("pasos_por_fruta", "mean"),
        generados=("nodos_generados_prom", "mean"),
        expandidos=("nodos_expandidos_prom", "mean"),
        frontera=("max_frontera_prom", "mean"),
        tiempo_ms=("tiempo_decision_ms", "mean"),
        n=("semilla", "count"),
    ).reset_index()
    return _limpiar({"disponible": True, "filas": res.to_dict(orient="records")})


RUTAS_POST = {
    "/api/instancia": api_instancia,
    "/api/buscar": api_buscar,
    "/api/comparar": api_comparar,
    "/api/partida": api_partida,
    "/api/jugar/nuevo": api_jugar_nuevo,
    "/api/jugar/paso": api_jugar_paso,
    "/api/jugar/pista": api_jugar_pista,
    "/api/benchmark": api_benchmark,
}

TIPOS = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
}


# =============================================================================
# SERVIDOR HTTP
# =============================================================================

class _Manejador(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silencia el log en la celda del notebook
        pass

    def _enviar(self, codigo: int, cuerpo: bytes, tipo: str) -> None:
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(cuerpo)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(cuerpo)

    def _json(self, codigo: int, datos: Any) -> None:
        self._enviar(codigo, json.dumps(datos, ensure_ascii=False).encode("utf-8"),
                     "application/json; charset=utf-8")

    def do_GET(self):
        ruta = self.path.split("?")[0]
        if ruta.startswith("/api/"):
            return self._atender_api(ruta, {})
        nombre = "index.html" if ruta in ("", "/") else ruta.lstrip("/")
        archivo = (CARPETA_WEB / nombre).resolve()
        if CARPETA_WEB.resolve() not in archivo.parents or not archivo.is_file():
            return self._enviar(404, b"No encontrado", "text/plain")
        self._enviar(200, archivo.read_bytes(), TIPOS.get(archivo.suffix, "application/octet-stream"))

    def do_POST(self):
        largo = int(self.headers.get("Content-Length", 0))
        try:
            datos = json.loads(self.rfile.read(largo) or b"{}")
        except json.JSONDecodeError:
            return self._json(400, {"error": "JSON inválido"})
        self._atender_api(self.path.split("?")[0], datos)

    def _atender_api(self, ruta: str, datos: Dict[str, Any]) -> None:
        funcion = RUTAS_POST.get(ruta)
        if funcion is None:
            return self._json(404, {"error": f"Ruta {ruta} no existe"})
        try:
            self._json(200, funcion(datos))
        except Exception as e:  # el error se muestra en la página
            self._json(500, {"error": f"{type(e).__name__}: {e}"})


# El servidor se guarda en ``sys`` y no en este módulo: así, si el notebook hace
# importlib.reload(interfaz), se apaga el servidor viejo y el nuevo usa el código actualizado.
import sys as _sys


def _puerto_libre(preferido: int) -> int:
    for puerto in range(preferido, preferido + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", puerto)) != 0:
                return puerto
    raise RuntimeError("No hay puertos libres.")


def iniciar_servidor(puerto: int = 8050) -> int:
    """
    Levanta el servidor en un hilo aparte. Si ya había uno de una carga anterior
    del módulo, lo apaga y reutiliza su puerto (la pestaña abierta sigue sirviendo).
    """
    previo = getattr(_sys, "_snakelab_servidor", None)
    if previo is not None:
        if previo[0].RequestHandlerClass is _Manejador:
            return previo[1]
        puerto = previo[1]
        detener_servidor()
    libre = _puerto_libre(puerto)
    servidor = ThreadingHTTPServer(("0.0.0.0", libre), _Manejador)
    servidor.daemon_threads = True
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    _sys._snakelab_servidor = (servidor, libre)
    return libre


def detener_servidor() -> None:
    """Apaga el servidor (por ejemplo, antes de cerrar el notebook)."""
    previo = getattr(_sys, "_snakelab_servidor", None)
    if previo is not None:
        previo[0].shutdown()
        previo[0].server_close()
        _sys._snakelab_servidor = None


def lanzar_aplicacion_snake(puerto: int = 8050, modo: str = "pestana") -> None:
    """
    Abre SnakeLab Studio.
    - En Google Colab: muestra un enlace que abre la página en otra pestaña
      (modo="pestana") o la incrusta en la celda (modo="iframe").
    - En Jupyter/VS Code local: muestra el enlace a http://localhost:<puerto>.
    """
    from IPython.display import HTML, display

    p = iniciar_servidor(puerto)
    try:
        from google.colab import output  # type: ignore

        if modo == "iframe":
            output.serve_kernel_port_as_iframe(p, height=900)
        else:
            output.serve_kernel_port_as_window(p, anchor_text="🐍 Abrir SnakeLab Studio en otra pestaña")
    except ImportError:
        url = f"http://localhost:{p}"
        display(HTML(
            f'<a href="{url}" target="_blank" style="font-size:15px;font-weight:600;'
            f'color:#16a34a">🐍 Abrir SnakeLab Studio en otra pestaña ({url})</a>'
        ))


if __name__ == "__main__":
    import time

    p = iniciar_servidor()
    print(f"SnakeLab Studio en http://localhost:{p}  (Ctrl+C para salir)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        detener_servidor()
