"""Utilidades de simulación, evaluación experimental y visualización.

El módulo contiene:

1. Clases para simular y comparar agentes: ``Episode``, ``Plot``,
   ``Experiment``, ``EnvfromGameAndPl2`` y ``ExperimentEspacios``.
2. El benchmark reproducible sobre N semillas:
   ``evaluar_configuracion_en_semillas``,
   ``ejecutar_benchmark_completo`` y ``graficar_resultados_benchmark``.
3. Dos paneles interactivos en ipywidgets,
   ``panel_inspector_busqueda`` y ``panel_simulador_partida``. La
   aplicación principal es SnakeLab Studio, en ``interfaz.py``; estos
   paneles se conservan como alternativa dentro del notebook.
"""

import ast
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
from time import sleep, time
from typing import Any, List, Mapping, Optional, Union, Dict, Tuple

from IPython.display import clear_output, display
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn import histplot, lineplot
from tqdm.auto import tqdm

from ambientes import (
    EstadoSnake,
    Snake,
    SubproblemaSnake,
    SnakeJuegoAdversario,
    dibujar_tablero_snake,
    obtener_acciones_aplicables,
)
from busqueda import (
    CATALOGO_HEURISTICAS,
    comparar_algoritmos_en_instancia,
    resolver_subproblema,
    solucion,
)
from agentes import AgenteSnake, AgenteFrutaAdversaria

# =============================================================================
# 1. Simulación y comparación de agentes
# =============================================================================


class Episode:
    """Simula un agente en un entorno y registra cada ronda.

    Parameters
    ----------
    environment : Snake
        Entorno con ``reset``, ``step`` y ``render``.
    agent : Agent or None
        Agente que juega; puede asignarse después.
    model_name : str
        Nombre del modelo en los datos registrados.
    num_rounds : int
        Máximo de rondas por episodio.
    id : int
        Identificador del episodio.
    """

    def __init__(
        self, environment, agent, model_name: str, num_rounds: int, id: int = 0
    ):
        """Crea el episodio y reinicia el entorno y el agente."""
        self.environment = environment
        self.agent = agent
        self.model_name = model_name
        self.num_rounds = num_rounds
        self.done = False
        self.T = 1
        self.id = id
        self.sleep_time = 0.25
        state = self.environment.reset()
        self.initial_state = state
        self.state = state
        if agent is not None:
            self.agent.restart()
            self.agent.states.append(state)

    def play_round(self, verbose: int = 0):
        """Juega una ronda: el agente decide y el entorno ejecuta.

        Con ``verbose > 3`` imprime el detalle de la ronda.
        """
        # Los agentes aleatorios eligen solo entre acciones aplicables.
        if hasattr(self.agent, "choices"):
            try:
                possible_actions = self.environment.acciones_aplicables(
                    self.state
                )
                if possible_actions:
                    self.agent.choices = possible_actions
            except Exception:
                pass

        action = self.agent.make_decision()
        self.agent.actions.append(action)
        next_state, reward, done = self.environment.step(action)

        if verbose > 3:
            state = self.agent.states[-1]
            print(f"\tThe state is => {state}")
            print(f"\tAgent takes action => {action}")
            print(f"\tThe state obtained is => {next_state}")
            print(f"\tThe reward obtained is => {reward}")
            print(f"\tEnvironment is finished? => {done}")

        self.agent.states.append(next_state)
        self.agent.rewards.append(reward)
        self.agent.dones.append(done)
        self.state = next_state
        self.T += 1
        self.done = done

    def run(self, verbose: int = 0) -> pd.DataFrame:
        """Juega rondas hasta terminar o llegar a ``num_rounds``.

        Returns
        -------
        pandas.DataFrame
            Una fila por ronda, como la devuelve ``to_pandas``.
        """
        for r in range(self.num_rounds):
            if not self.done:
                if verbose > 2:
                    print("\n" + "-" * 10 + f"Round {r}" + "-" * 10 + "\n")
                self.play_round(verbose=verbose)
            else:
                break
        return self.to_pandas()

    def to_pandas(self) -> pd.DataFrame:
        """Devuelve el historial del episodio como un DataFrame."""
        # La última ronda no tiene acción asociada.
        self.agent.actions.append(np.nan)
        data = {
            "episode": [self.id] * self.T,
            "round": list(range(self.T)),
            "state": self.agent.states[: self.T],
            "action": self.agent.actions[: self.T],
            "reward": self.agent.rewards[: self.T],
            "done": self.agent.dones[: self.T],
        }
        df = pd.DataFrame.from_dict(data)
        df["model"] = self.model_name
        return df

    def reset(self):
        """Reinicia el entorno, el agente y los contadores."""
        state = self.environment.reset()
        self.initial_state = state
        self.state = state
        if self.agent is not None:
            self.agent.restart()
            self.agent.states.append(state)
        self.T = 1
        self.done = False

    def renderize(self):
        """Juega el episodio dibujando el tablero en cada ronda."""
        for _ in range(self.num_rounds):
            if not self.done:
                self.play_round(verbose=0)
                clear_output(wait=True)
                self.environment.render()
                sleep(self.sleep_time)
            else:
                break

    def simulate(
        self, num_episodes: int = 1, file: str = None, verbose: int = 0
    ) -> pd.DataFrame:
        """Juega varios episodios, uno por semilla, y junta sus datos.

        Parameters
        ----------
        num_episodes : int
            Número de episodios; el episodio ``i`` usa la semilla ``i``.
        file : str, optional
            Ruta de un CSV donde guardar los datos.
        verbose : int
            Nivel de detalle de los mensajes.
        """
        data_frames = []
        for ep in range(num_episodes):
            if verbose > 1:
                print("\n" + "=" * 10 + f"Episode {ep}" + "=" * 10 + "\n")
            if hasattr(self.environment, "semilla"):
                self.environment.semilla = ep
            self.reset()
            self.id = ep
            df = self.run(verbose=verbose)
            data_frames.append(df)
        data = pd.concat(data_frames, ignore_index=True)
        if file is not None:
            data.to_csv(file, index=False)
        return data


class Plot:
    """Gráficas frecuentes de los datos de ``Episode`` o ``Experiment``.

    Cada método acepta ``file`` para guardar la figura.
    """

    def __init__(self, data: pd.DataFrame):
        """Guarda los datos a graficar."""
        self.data = data

    def plot_rewards(self, file: str = None):
        """Grafica la recompensa total de cada episodio."""
        models = self.data.model.unique()
        vs_models = len(models) > 1
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        data = self.data.copy()
        if "simulation" in data.columns:
            data = (
                data.groupby(
                    ["model", "environment", "simulation", "episode"]
                )["reward"]
                .sum()
                .reset_index()
            )
        else:
            data = (
                data.groupby(["model", "episode"])["reward"]
                .sum()
                .reset_index()
            )
        ax.set_xlabel("Episodio (Semilla)")
        ax.set_ylabel("Recompensa total (Ganancia - Pérdida)")
        ax.grid(alpha=0.3)
        if vs_models:
            ax = lineplot(x="episode", y="reward", hue="model", data=data)
        else:
            ax = lineplot(x="episode", y="reward", data=data)
        if file is not None:
            plt.savefig(file, dpi=300, bbox_inches="tight")
        return ax

    def plot_round_reward(self, file: str = None):
        """Grafica la recompensa promedio en cada ronda."""
        models = self.data.model.unique()
        vs_models = len(models) > 1
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.set_xlabel("Paso (Round)")
        ax.set_ylabel("Recompensa promedio")
        ax.grid(alpha=0.3)
        if vs_models:
            ax = lineplot(x="round", y="reward", hue="model", data=self.data)
        else:
            ax = lineplot(x="round", y="reward", data=self.data)
        if file is not None:
            plt.savefig(file, dpi=300, bbox_inches="tight")
        return ax

    def plot_histogram_rewards(self, file: str = None):
        """Grafica el histograma de la recompensa total por partida."""
        models = self.data.model.unique()
        vs_models = len(models) > 1
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.set_xlabel("Suma de recompensas por partida")
        ax.set_ylabel("Frecuencia")
        ax.grid(alpha=0.3)
        columnas = ["model", "episode"]
        if "simulation" in self.data.columns:
            columnas.append("simulation")
        if "environment" in self.data.columns:
            columnas.append("environment")
        df = (
            self.data.groupby(columnas)
            .agg({"reward": "sum", "done": "sum"})
            .reset_index()
        )
        if vs_models:
            ax = histplot(x="reward", hue="model", data=df, kde=True)
        else:
            ax = histplot(x="reward", data=df, kde=True)
        if file is not None:
            plt.savefig(file, dpi=300, bbox_inches="tight")
        return ax


class Experiment:
    """Compara varios agentes sobre un conjunto de entornos.

    Parameters
    ----------
    num_rounds : int
        Máximo de rondas por episodio.
    num_episodes : int
        Episodios por simulación; el episodio ``i`` usa la semilla
        ``i``.
    num_simulations : int
        Número de repeticiones del experimento.
    """

    def __init__(
        self, num_rounds: int, num_episodes: int, num_simulations: int = 1
    ):
        """Crea el experimento sin entornos cargados."""
        self.num_rounds = num_rounds
        self.num_episodes = num_episodes
        self.num_simulations = num_simulations
        self.test_suite = []
        self.test_suite_names = []
        self.data = None

    def load_env(self, environment, env_name: str = "SnakeEnv"):
        """Usa ``environment`` como único entorno del experimento."""
        self.test_suite = [environment]
        self.test_suite_names = [env_name]

    def run_experiment(
        self, agents: list, names: list, measures: list, verbose: int = 0
    ):
        """Juega todos los agentes en todos los entornos y grafica.

        Parameters
        ----------
        agents : list
            Agentes a comparar.
        names : list of str
            Nombre de cada agente.
        measures : list of str
            Gráficas a mostrar: ``"reward"``, ``"round_reward"`` o
            ``"histogram"``.
        verbose : int
            Nivel de detalle de los mensajes.
        """
        assert len(self.test_suite) > 0, "Error: Test suite cannot be empty!"
        data_frames = []
        for counter, environment in enumerate(
            tqdm(self.test_suite, desc="Evaluando entornos")
        ):
            for k in range(self.num_simulations):
                for ep in range(self.num_episodes):
                    if hasattr(environment, "semilla"):
                        environment.semilla = ep
                    sim_core = Episode(
                        environment=environment,
                        agent=None,
                        model_name=None,
                        num_rounds=self.num_rounds,
                        id=ep,
                    )
                    # Cada agente juega una copia del mismo episodio.
                    for idx_ag, agent in enumerate(agents):
                        sim = deepcopy(sim_core)
                        agent.restart()
                        sim.agent = agent
                        sim.agent.states.append(sim.initial_state)
                        sim.model_name = names[idx_ag]
                        df = sim.run(verbose=verbose)
                        df["environment"] = self.test_suite_names[counter]
                        df["simulation"] = k
                        df["agent"] = names[idx_ag]
                        data_frames.append(df)
        self.data = pd.concat(data_frames, ignore_index=True)
        for m in measures:
            if m == "reward":
                Plot(self.data).plot_rewards()
            elif m == "round_reward":
                Plot(self.data).plot_round_reward()
            elif m == "histogram":
                Plot(self.data).plot_histogram_rewards()
            plt.show()


class EnvfromGameAndPl2:
    """Convierte un juego de dos jugadores en un entorno de un agente.

    El agente controla a la serpiente; cada vez que el turno pasa al
    otro jugador, este responde de inmediato dentro de ``step``.

    Parameters
    ----------
    game : SnakeJuegoAdversario
        Juego de dos jugadores.
    other_player : Agent
        Agente que juega como la fruta.
    """

    def __init__(self, game: Any, other_player: Any):
        """Guarda el juego y una copia de su estado inicial."""
        self.other_player = other_player
        self.initial_game = deepcopy(game)
        self.game = game
        self.state = self.game.estado_inicial

    def reset(self):
        """Reinicia juego y oponente y devuelve el estado inicial."""
        self.game = deepcopy(self.initial_game)
        self.state = self.game.estado_inicial
        self.other_player.reset()
        self.other_player.states.append(self.state)
        return self.state

    def render(self, ax=None, titulo=None):
        """Dibuja el estado actual del juego."""
        dibujar_tablero_snake(self.state, ax=ax, titulo=titulo)

    def acciones_aplicables(self, state):
        """Devuelve las jugadas posibles en ``state``."""
        return self.game.acciones(state)

    def step(self, action):
        """Aplica la acción de la serpiente y la respuesta del oponente.

        Returns
        -------
        tuple
            ``(nuevo_estado, recompensa, done)``.
        """
        state = self.state
        playing = self.game.player(state)
        new_state = self.game.resultado(state, action)
        reward = self.game.utilidad(new_state, playing)
        reward = reward if reward is not None else 0.0
        done = self.game.es_terminal(new_state)

        # Si el turno pasó a la fruta, el oponente juega de inmediato.
        if not done and self.game.player(new_state) != "serpiente":
            self.other_player.states.append(new_state)
            other_action = self.other_player.make_decision()
            new_state = self.game.resultado(new_state, other_action)
            reward = self.game.utilidad(new_state, "serpiente")
            reward = reward if reward is not None else 0.0
            done = self.game.es_terminal(new_state)

        self.state = new_state
        return new_state, reward, done


class ExperimentEspacios:
    """Compara funciones de búsqueda sobre varios espacios de estados.

    Mide tiempo de CPU, número de pasos, costo del camino y nodos
    generados y expandidos de cada función en cada espacio.
    """

    def __init__(self):
        """Crea el experimento sin espacios cargados."""
        self.test_suite = []
        self.test_suite_names = []
        self.test_suite_classes = []
        self.test_suite_pasos = []
        self.data = None
        self.tabla = None

    def load_space(
        self,
        space,
        env_name: str = "SubproblemaSnake",
        class_name: str = "SubproblemaSnake",
        step: int = 1,
    ):
        """Agrega un espacio de estados al experimento.

        Parameters
        ----------
        space : SubproblemaSnake
            Problema sobre el que se ejecutan las funciones.
        env_name, class_name : str
            Nombres para identificar el espacio en la tabla.
        step : int
            Número de pasos esperado de la solución.
        """
        self.test_suite.append([space])
        self.test_suite_names.append(env_name)
        self.test_suite_classes.append(class_name)
        self.test_suite_pasos.append(step)

    def obtiene_tiempos(self, fun, args: list, num_it: int = 5) -> tuple:
        """Ejecuta ``fun(*args)`` ``num_it`` veces y mide cada una.

        Returns
        -------
        tuple
            ``(tiempos_en_ms, soluciones)``.
        """
        tiempos_fun = []
        soluciones = []
        for _ in range(num_it):
            arranca = time()
            x = fun(*args)
            para = time()
            tiempos_fun.append((para - arranca) * 1000.0)  # En ms.
            soluciones.append(x)
        return tiempos_fun, soluciones

    def compara_funciones(
        self, lista_funs: list, lista_nombres: list, num_it: int = 5
    ) -> pd.DataFrame:
        """Ejecuta cada función en cada espacio y tabula sus métricas.

        Returns
        -------
        pandas.DataFrame
            Una fila por función, espacio y repetición.
        """
        registros = []
        for idx_s, state in enumerate(self.test_suite):
            for idx_f, fun in enumerate(lista_funs):
                ts, nodos = self.obtiene_tiempos(fun, state, num_it)
                for t_val, n_sol in zip(ts, nodos):
                    accs = solucion(n_sol) if n_sol is not None else None
                    registros.append(
                        {
                            "Función": lista_nombres[idx_f],
                            "tiempo_CPU_ms": t_val,
                            "Espacio": self.test_suite_classes[idx_s],
                            "Nombre": self.test_suite_names[idx_s],
                            "Paso_esperado": self.test_suite_pasos[idx_s],
                            "Num_pasos": (
                                len(accs) if accs is not None else np.nan
                            ),
                            "Costo_camino": (
                                n_sol.costo_camino
                                if n_sol is not None
                                else np.nan
                            ),
                            "Nodos_generados": getattr(
                                n_sol, "nodos_generados", np.nan
                            ),
                            "Nodos_expandidos": getattr(
                                n_sol, "nodos_expandidos", np.nan
                            ),
                            "Max_frontera": getattr(
                                n_sol, "max_frontera", np.nan
                            ),
                        }
                    )
        self.data = pd.DataFrame(registros)
        return self.data


# =============================================================================
# 2. Evaluación experimental sobre N semillas
# =============================================================================


def evaluar_configuracion_en_semillas(
    algoritmo: str,
    heuristica: str = "manhattan",
    chequeo_seguridad: bool = True,
    lambda_riesgo: float = 0.0,
    k_haz: int = 3,
    semillas: range = range(100),
    tamano: int = 10,
    k_max: int = 200,
    max_pasos_partida: int = 2500,
    mostrar_progreso: bool = True,
) -> List[Dict[str, Any]]:
    """Juega una partida por semilla con una configuración del agente.

    Parameters
    ----------
    algoritmo, heuristica : str
        Algoritmo de búsqueda y heurística del agente.
    chequeo_seguridad : bool
        Si el agente usa el chequeo de seguridad.
    lambda_riesgo : float
        Peso del riesgo en el costo por paso.
    k_haz : int
        Ancho del haz para la búsqueda en haz.
    semillas : range
        Semillas a jugar, por defecto 0 a 99.
    tamano, k_max : int
        Lado del tablero y máximo de pasos sin comer.
    max_pasos_partida : int
        Tope de pasos por partida.
    mostrar_progreso : bool
        Si se muestra una barra de progreso.

    Returns
    -------
    list of dict
        Una fila por partida con sus métricas y su causa de fin.
    """
    resultados_partidas: List[Dict[str, Any]] = []
    iterador = tqdm(
        semillas,
        desc=f"{algoritmo} ({heuristica}) | Seg={chequeo_seguridad}",
        leave=False,
        disable=not mostrar_progreso,
    )

    for sem in iterador:
        env = Snake(
            tamano=tamano,
            semilla=int(sem),
            lambda_riesgo=lambda_riesgo,
            k_max=k_max,
        )
        agente = AgenteSnake(
            algoritmo=algoritmo,
            heuristica=heuristica,
            chequeo_seguridad=chequeo_seguridad,
            lambda_riesgo=lambda_riesgo,
            k_haz=k_haz,
        )
        estado = env.reset(semilla=int(sem))
        agente.restart()
        agente.states.append(estado)

        while not env.done and env.pasos < max_pasos_partida:
            accion = agente.make_decision()
            estado, reward, done = env.step(accion)
            agente.states.append(estado)
            agente.rewards.append(reward)
            agente.dones.append(done)

        # Métricas de la partida, promediando las de cada decisión.
        df_dec = pd.DataFrame(agente.historial_decisiones)
        victoria = env.test_objetivo(env.estado)
        frutas = env.frutas_comidas
        pasos_por_fruta = (env.pasos / frutas) if frutas > 0 else np.nan

        resultados_partidas.append(
            {
                "configuracion": f"{algoritmo}"
                + (f" ({heuristica})" if heuristica != "ninguna" else "")
                + (" + Seg" if chequeo_seguridad else " (Sin Seg)"),
                "algoritmo": algoritmo,
                "heuristica": heuristica,
                "chequeo_seguridad": chequeo_seguridad,
                "lambda_riesgo": lambda_riesgo,
                "semilla": int(sem),
                "longitud_final": env.estado.longitud,
                "frutas_comidas": frutas,
                "victoria": int(victoria),
                "pasos_totales": env.pasos,
                "pasos_por_fruta": pasos_por_fruta,
                "recompensa_neta": env.ganancia_total - env.perdida_total,
                "ganancia_total": env.ganancia_total,
                "perdida_total": env.perdida_total,
                "nodos_generados_prom": (
                    df_dec["nodos_generados"].mean()
                    if not df_dec.empty
                    else 0.0
                ),
                "nodos_expandidos_prom": (
                    df_dec["nodos_expandidos"].mean()
                    if not df_dec.empty
                    else 0.0
                ),
                "max_frontera_prom": (
                    df_dec["max_frontera"].max() if not df_dec.empty else 0
                ),
                "tiempo_decision_ms": (
                    df_dec["tiempo_ms"].mean() if not df_dec.empty else 0.0
                ),
                "evasiones_cola": agente.pasos_siguiendo_cola,
                "causa_fin": env.causa_fin,
            }
        )
    return resultados_partidas


def ejecutar_benchmark_completo(
    num_semillas: int = 100,
    tamano: int = 10,
    k_max: int = 200,
    archivo_csv: Optional[str] = "resultados_benchmark_snake.csv",
    forzar_recalculo: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta el benchmark completo o lo carga desde un CSV.

    Juega las 14 configuraciones del experimento, los seis algoritmos
    con y sin chequeo de seguridad más las variantes de heurística y de
    lambda, sobre las semillas 0 a ``num_semillas - 1``. Si
    ``archivo_csv`` existe y ``forzar_recalculo`` es falso, carga los
    resultados guardados en lugar de jugar.

    Returns
    -------
    tuple of pandas.DataFrame
        ``(datos_crudos, resumen_por_configuracion)``.
    """
    if archivo_csv and os.path.exists(archivo_csv) and not forzar_recalculo:
        df_crudo = pd.read_csv(archivo_csv)
    else:
        experimentos = [
            # (algoritmo, heurística, chequeo, lambda, k del haz).
            # Los seis algoritmos sin chequeo de seguridad.
            ("BFS", "ninguna", False, 0.0, 3),
            ("DFS", "ninguna", False, 0.0, 3),
            ("UCS", "ninguna", False, 0.5, 3),
            ("Avara", "manhattan", False, 0.0, 3),
            ("A*", "manhattan", False, 0.0, 3),
            ("Beam", "manhattan", False, 0.0, 3),
            # Los seis algoritmos con chequeo, más A* con h_2 y lambda
            # 0.5.
            ("BFS", "ninguna", True, 0.0, 3),
            ("DFS", "ninguna", True, 0.0, 3),
            ("UCS", "ninguna", True, 0.5, 3),
            ("Avara", "manhattan", True, 0.0, 3),
            ("A*", "manhattan", True, 0.0, 3),
            ("A*", "euclidea", True, 0.0, 3),
            ("A*", "manhattan", True, 0.5, 3),
            ("Beam", "manhattan", True, 0.0, 3),
        ]
        todas_las_filas: List[Dict[str, Any]] = []
        for alg, heur, seg, lam, kh in tqdm(
            experimentos, desc="Progreso global Benchmark"
        ):
            filas = evaluar_configuracion_en_semillas(
                algoritmo=alg,
                heuristica=heur,
                chequeo_seguridad=seg,
                lambda_riesgo=lam,
                k_haz=kh,
                semillas=range(num_semillas),
                tamano=tamano,
                k_max=k_max,
            )
            todas_las_filas.extend(filas)

        df_crudo = pd.DataFrame(todas_las_filas)
        if archivo_csv:
            df_crudo.to_csv(archivo_csv, index=False)

    resumen = (
        df_crudo.groupby(
            ["algoritmo", "heuristica", "chequeo_seguridad", "lambda_riesgo"],
            as_index=False,
        )
        .agg(
            Longitud_Prom=("longitud_final", "mean"),
            Longitud_Std=("longitud_final", "std"),
            Longitud_Max=("longitud_final", "max"),
            Victorias_Pct=("victoria", lambda s: round(100.0 * s.mean(), 1)),
            Pasos_por_Fruta=("pasos_por_fruta", "mean"),
            Nodos_Gen_Prom=("nodos_generados_prom", "mean"),
            Nodos_Exp_Prom=("nodos_expandidos_prom", "mean"),
            Max_Frontera=("max_frontera_prom", "mean"),
            Tiempo_Dec_ms=("tiempo_decision_ms", "mean"),
            Recompensa_Prom=("recompensa_neta", "mean"),
        )
        .round(2)
    )
    return df_crudo, resumen


def graficar_resultados_benchmark(df_crudo: pd.DataFrame):
    """Dibuja las cuatro gráficas comparativas del benchmark.

    Muestra la longitud final con y sin chequeo, los nodos expandidos
    por decisión, la distribución de la recompensa neta y la relación
    entre frontera máxima y tiempo por decisión.
    """
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 10), gridspec_kw={"hspace": 0.35, "wspace": 0.25}
    )

    # 1. Longitud final con vs sin chequeo de seguridad
    df_principales = df_crudo[df_crudo["lambda_riesgo"] == 0.0].copy()
    sns.barplot(
        data=df_principales,
        x="algoritmo",
        y="longitud_final",
        hue="chequeo_seguridad",
        ax=axes[0, 0],
        palette=["#fc8d59", "#2b8cbe"],
    )
    axes[0, 0].set_title(
        "1. Longitud Final Promedio (Con vs. Sin Chequeo de Seguridad)",
        weight="bold",
    )
    axes[0, 0].set_xlabel("Algoritmo")
    axes[0, 0].set_ylabel("Longitud alcanzada (L_final)")
    axes[0, 0].grid(axis="y", alpha=0.3)

    # 2. Nodos expandidos por decisión
    df_seg = df_crudo[df_crudo["chequeo_seguridad"].eq(True)].copy()
    df_seg["Etiqueta"] = (
        df_seg["algoritmo"] + "\n(" + df_seg["heuristica"] + ")"
    )
    sns.barplot(
        data=df_seg,
        x="Etiqueta",
        y="nodos_expandidos_prom",
        hue="Etiqueta",
        legend=False,
        ax=axes[0, 1],
        palette="viridis",
    )
    axes[0, 1].set_title(
        "2. Nodos Expandidos Promedio por Decisión (Con Seguridad)",
        weight="bold",
    )
    axes[0, 1].set_xlabel("Algoritmo y Heurística")
    axes[0, 1].set_ylabel("Nodos expandidos")
    axes[0, 1].tick_params(axis="x", rotation=25)
    axes[0, 1].grid(axis="y", alpha=0.3)

    # 3. Distribución de recompensa neta (Ganancia - Pérdida)
    sns.boxplot(
        data=df_principales[df_principales["chequeo_seguridad"].eq(True)],
        x="algoritmo",
        y="recompensa_neta",
        hue="algoritmo",
        legend=False,
        ax=axes[1, 0],
        palette="Set2",
    )
    axes[1, 0].set_title(
        "3. Distribución de Recompensa Neta (Ganancia - Pérdida)",
        weight="bold",
    )
    axes[1, 0].set_xlabel("Algoritmo")
    axes[1, 0].set_ylabel("Recompensa Acumulada")
    axes[1, 0].grid(axis="y", alpha=0.3)

    # 4. Tiempo de cómputo vs Tamaño máximo de frontera
    sns.scatterplot(
        data=df_seg.groupby(["algoritmo", "heuristica"], as_index=False).mean(
            numeric_only=True
        ),
        x="max_frontera_prom",
        y="tiempo_decision_ms",
        hue="algoritmo",
        style="heuristica",
        s=140,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title(
        "4. Compromiso Memoria (Máx Frontera) vs. Tiempo por Decisión (ms)",
        weight="bold",
    )
    axes[1, 1].set_xlabel("Tamaño máximo de la frontera (Memoria)")
    axes[1, 1].set_ylabel("Tiempo promedio por decisión (ms)")
    axes[1, 1].grid(alpha=0.3)

    plt.show()


# =============================================================================
# 3. Paneles interactivos con ipywidgets
# =============================================================================


def panel_inspector_busqueda():
    """Muestra un panel para reproducir una búsqueda paso a paso.

    Permite elegir algoritmo, heurística e instancia, y avanzar por las
    iteraciones con controles de reproducción. El tablero muestra nodos
    expandidos en azul, frontera en amarillo, nodo actual en naranja y
    camino en verde; un recuadro muestra la iteración, el tamaño de la
    frontera y los valores g(n), h(n) y f(n).
    """
    w_alg = widgets.Dropdown(
        options=["BFS", "DFS", "UCS", "Avara", "A*", "Beam"],
        value="A*",
        description="Algoritmo:",
        layout=widgets.Layout(width="210px"),
    )
    w_heur = widgets.Dropdown(
        options=["manhattan", "euclidea", "riesgo"],
        value="manhattan",
        description="Heurística:",
        layout=widgets.Layout(width="220px"),
    )
    w_semilla = widgets.IntSlider(
        value=12,
        min=0,
        max=99,
        step=1,
        description="Semilla:",
        layout=widgets.Layout(width="260px"),
    )
    w_pasos_previos = widgets.IntSlider(
        value=18,
        min=0,
        max=60,
        step=2,
        description="Madurez (L):",
        layout=widgets.Layout(width="270px"),
    )
    w_obs = widgets.IntSlider(
        value=6,
        min=0,
        max=20,
        step=1,
        description="Obstáculos:",
        layout=widgets.Layout(width="250px"),
    )
    w_riesgo = widgets.FloatSlider(
        value=0.5,
        min=0.0,
        max=3.0,
        step=0.25,
        description="Riesgo λ:",
        layout=widgets.Layout(width="250px"),
    )
    w_khaz = widgets.IntSlider(
        value=3,
        min=1,
        max=10,
        step=1,
        description="Ancho Haz k:",
        layout=widgets.Layout(width="250px"),
    )
    w_toro = widgets.Checkbox(
        value=False, description="Tablero Toroidal", indent=False
    )

    play = widgets.Play(
        value=0,
        min=0,
        max=10,
        step=1,
        interval=350,
        description="Reproducir",
        disabled=False,
    )
    slider_paso = widgets.IntSlider(
        value=0,
        min=0,
        max=10,
        step=1,
        description="Iteración:",
        layout=widgets.Layout(width="420px"),
    )
    widgets.jslink((play, "value"), (slider_paso, "value"))

    btn_ant = widgets.Button(
        description="◀ Paso -1", layout=widgets.Layout(width="95px")
    )
    btn_sig = widgets.Button(
        description="Paso +1 ▶", layout=widgets.Layout(width="95px")
    )
    btn_recalcular = widgets.Button(
        description="🔄 Generar y Resolver Instancia",
        button_style="primary",
        layout=widgets.Layout(width="240px"),
    )

    panel_estado_html = widgets.HTML()
    salida_grafica = widgets.Output()

    estado_cache: Dict[str, Any] = {"estado": None, "resultado": None}

    def construir_instancia(*_):
        """Genera la instancia elegida y la resuelve con traza."""
        env = Snake(
            tamano=10,
            semilla=w_semilla.value,
            toroidal=w_toro.value,
            num_obstaculos=w_obs.value,
            lambda_riesgo=w_riesgo.value,
        )
        ag = AgenteSnake(
            algoritmo="A*", heuristica="manhattan", chequeo_seguridad=True
        )
        est = env.reset(semilla=w_semilla.value)
        ag.restart()
        ag.states.append(est)

        # Se juegan algunos turnos para que la serpiente tenga cuerpo.
        for _ in range(w_pasos_previos.value):
            if env.done:
                break
            acc = ag.make_decision()
            est, r, d = env.step(acc)
            ag.states.append(est)
            ag.rewards.append(r)
            ag.dones.append(d)

        if env.done or env.estado.fruta is None:
            env.reset(semilla=w_semilla.value)

        prob = SubproblemaSnake(
            estado_inicial=env.estado,
            objetivo=env.estado.fruta,
            lambda_riesgo=w_riesgo.value,
        )
        res = resolver_subproblema(
            prob,
            algoritmo=w_alg.value,
            heuristica=w_heur.value,
            k_haz=w_khaz.value,
            guardar_traza=True,
        )
        estado_cache["estado"] = env.estado
        estado_cache["resultado"] = res

        max_idx = max(0, len(res.traza))
        play.max = max_idx
        slider_paso.max = max_idx
        slider_paso.value = max_idx  # Por defecto, el resultado final.
        dibujar_iteracion()

    def dibujar_iteracion(*_):
        """Dibuja el tablero y el estado interno de la iteración."""
        est: EstadoSnake = estado_cache["estado"]
        res = estado_cache["resultado"]
        if est is None or res is None:
            return

        idx = slider_paso.value
        es_vista_final = idx == len(res.traza)

        if len(res.traza) == 0:
            frontera_c, expandidos_c, camino_c, actual_c = (
                [],
                [],
                [],
                est.cabeza,
            )
            g_val, h_val, f_val, tam_f = 0.0, 0.0, 0.0, 0
        elif es_vista_final:
            ult = res.traza[-1]
            frontera_c = ult.frontera_celdas
            expandidos_c = ult.expandidos_celdas
            camino_c = res.camino_celdas if res.exito else ult.camino_parcial
            actual_c = ult.celda_actual
            g_val, h_val, f_val, tam_f = (
                ult.g,
                ult.h,
                ult.f,
                ult.tamano_frontera,
            )
        else:
            paso = res.traza[idx]
            frontera_c = paso.frontera_celdas
            expandidos_c = paso.expandidos_celdas
            camino_c = paso.camino_parcial
            actual_c = paso.celda_actual
            g_val, h_val, f_val, tam_f = (
                paso.g,
                paso.h,
                paso.f,
                paso.tamano_frontera,
            )

        etiqueta_final = (
            "(SOLUCIÓN FINAL)" if es_vista_final and res.exito else ""
        )
        panel_estado_html.value = f"""
        <div style="background:#f1f3f5; padding:10px 14px;
                    border-radius:8px; border-left:5px solid #1c7ed6;
                    font-family:monospace; font-size:13px;">
            <b>ESTADO INTERNO DEL ALGORITMO
            ({res.algoritmo} | Heurística: {res.heuristica})</b><br>
            • <b>Iteración:</b> {idx}/{len(res.traza)} {etiqueta_final}
            | <b>Nodo actual:</b> {actual_c}
            | <b>Tamaño Frontera:</b> {tam_f}<br>
            • <b>Evaluación del Nodo:</b>
              Pérdida acumulada <code>g(n) = {g_val:.2f}</code> |
              Pérdida futura <code>h(n) = {h_val:.2f}</code> |
              <b>Total <code>f(n) = {f_val:.2f}</code></b><br>
            • <b>Métricas Globales:</b>
              Generados: <b>{res.nodos_generados}</b> |
              Expandidos: <b>{res.nodos_expandidos}</b> |
              Máx Frontera: <b>{res.max_frontera}</b> |
              Costo Solución: <b>{res.costo_total:.2f}</b>
              ({res.longitud_solucion} pasos) |
              Tiempo: <b>{res.tiempo_ms:.2f} ms</b>
        </div>
        """

        with salida_grafica:
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(6.2, 6.2))
            titulo_graf = (
                f"Proceso de Búsqueda: {res.algoritmo} ({res.heuristica})\n"
                "Azul=Expandidos | Amarillo=Frontera | "
                "Naranja=Nodo Actual | Verde=Camino"
            )
            dibujar_tablero_snake(
                est,
                camino=camino_c,
                frontera=frontera_c,
                expandidos=expandidos_c,
                celda_actual=actual_c,
                titulo=titulo_graf,
                ax=ax,
            )
            plt.show()

    def ir_anterior(_):
        """Retrocede una iteración."""
        if slider_paso.value > 0:
            slider_paso.value -= 1

    def ir_siguiente(_):
        """Avanza una iteración."""
        if slider_paso.value < slider_paso.max:
            slider_paso.value += 1

    btn_ant.on_click(ir_anterior)
    btn_sig.on_click(ir_siguiente)
    btn_recalcular.on_click(construir_instancia)
    slider_paso.observe(dibujar_iteracion, names="value")
    for w in (
        w_alg,
        w_heur,
        w_semilla,
        w_pasos_previos,
        w_obs,
        w_riesgo,
        w_khaz,
        w_toro,
    ):
        w.observe(construir_instancia, names="value")

    construir_instancia()

    controles_sup = widgets.HBox([w_alg, w_heur, w_semilla, w_pasos_previos])
    controles_med = widgets.HBox(
        [w_obs, w_riesgo, w_khaz, w_toro, btn_recalcular]
    )
    controles_rep = widgets.HBox([play, btn_ant, btn_sig, slider_paso])

    display(
        widgets.VBox(
            [
                controles_sup,
                controles_med,
                controles_rep,
                panel_estado_html,
                salida_grafica,
            ]
        )
    )


def panel_simulador_partida():
    """Muestra un panel para simular partidas completas o jugarlas.

    Incluye un editor de instancias, un reproductor turno a turno de una
    partida del agente y un modo manual para jugar con botones.
    """
    w_tam = widgets.Dropdown(
        options=[10, 15, 20],
        value=10,
        description="Tablero N:",
        layout=widgets.Layout(width="180px"),
    )
    w_alg = widgets.Dropdown(
        options=["BFS", "DFS", "UCS", "Avara", "A*", "Beam"],
        value="A*",
        description="Algoritmo:",
        layout=widgets.Layout(width="190px"),
    )
    w_heur = widgets.Dropdown(
        options=["manhattan", "euclidea", "riesgo"],
        value="manhattan",
        description="Heurística:",
        layout=widgets.Layout(width="210px"),
    )
    w_seg = widgets.Checkbox(
        value=True, description="Chequeo Seguridad (Cola)", indent=False
    )
    w_toro = widgets.Checkbox(
        value=False, description="Toroidal", indent=False
    )
    w_sem = widgets.IntSlider(
        value=0,
        min=0,
        max=99,
        description="Semilla:",
        layout=widgets.Layout(width="240px"),
    )
    w_obs = widgets.IntSlider(
        value=0,
        min=0,
        max=25,
        description="Obstáculos:",
        layout=widgets.Layout(width="240px"),
    )
    w_frutas = widgets.Dropdown(
        options=[1, 3],
        value=1,
        description="Frutas sim.:",
        layout=widgets.Layout(width="180px"),
    )
    w_pasos_sim = widgets.IntSlider(
        value=150,
        min=20,
        max=600,
        step=10,
        description="Pasos máx:",
        layout=widgets.Layout(width="250px"),
    )

    btn_simular = widgets.Button(
        description="▶ Simular Partida Agente",
        button_style="success",
        layout=widgets.Layout(width="210px"),
    )
    btn_reiniciar_manual = widgets.Button(
        description="🎮 Iniciar Modo Manual",
        button_style="warning",
        layout=widgets.Layout(width="190px"),
    )

    play_hist = widgets.Play(value=0, min=0, max=1, step=1, interval=180)
    slider_turno = widgets.IntSlider(
        value=0,
        min=0,
        max=1,
        step=1,
        description="Turno:",
        layout=widgets.Layout(width="420px"),
    )
    widgets.jslink((play_hist, "value"), (slider_turno, "value"))

    # Botones del modo manual.
    btn_n = widgets.Button(
        description="⬆ Norte", layout=widgets.Layout(width="90px")
    )
    btn_s = widgets.Button(
        description="⬇ Sur", layout=widgets.Layout(width="90px")
    )
    btn_o = widgets.Button(
        description="⬅ Oeste", layout=widgets.Layout(width="90px")
    )
    btn_e = widgets.Button(
        description="➡ Este", layout=widgets.Layout(width="90px")
    )
    caja_manual = widgets.HBox(
        [widgets.HTML("<b>Juego Manual:</b> "), btn_n, btn_s, btn_o, btn_e]
    )

    info_html = widgets.HTML()
    out_canvas = widgets.Output()

    memoria: Dict[str, Any] = {
        "frames": [],
        "env_manual": None,
        "modo": "auto",
    }

    def correr_simulacion_auto(_=None):
        """Juega una partida completa del agente y guarda cada turno."""
        memoria["modo"] = "auto"

        env = Snake(
            tamano=w_tam.value,
            semilla=w_sem.value,
            toroidal=w_toro.value,
            num_obstaculos=w_obs.value,
            num_frutas=w_frutas.value,
            k_max=200,
        )
        ag = AgenteSnake(
            algoritmo=w_alg.value,
            heuristica=w_heur.value,
            chequeo_seguridad=w_seg.value,
        )
        est = env.reset(semilla=w_sem.value)
        ag.restart()
        ag.states.append(est)

        frames = [
            {
                "estado": est,
                "paso": 0,
                "frutas": 0,
                "recompensa": 0.0,
                "causa": env.causa_fin,
                "plan": [],
            }
        ]
        recompensa_acum = 0.0
        while not env.done and env.pasos < w_pasos_sim.value:
            acc = ag.make_decision()
            est, rew, _ = env.step(acc)
            recompensa_acum += rew
            ag.states.append(est)
            ag.rewards.append(rew)
            ag.dones.append(env.done)
            frames.append(
                {
                    "estado": est,
                    "paso": env.pasos,
                    "frutas": env.frutas_comidas,
                    "recompensa": recompensa_acum,
                    "causa": env.causa_fin,
                    "accion": acc,
                }
            )

        memoria["frames"] = frames
        play_hist.max = len(frames) - 1
        slider_turno.max = len(frames) - 1
        slider_turno.value = 0
        dibujar_frame_auto()

    def dibujar_frame_auto(*_):
        """Dibuja el turno elegido de la partida automática."""
        if memoria["modo"] != "auto" or not memoria["frames"]:
            return
        f = memoria["frames"][slider_turno.value]
        est: EstadoSnake = f["estado"]
        ultimo_turno = len(memoria["frames"]) - 1
        info_html.value = f"""
        <div style="background:#e7f5ff; padding:8px 12px;
                    border-radius:6px; font-family:monospace;">
            <b>MODO AUTOMÁTICO ({w_alg.value})</b> |
            Turno: <b>{f['paso']}/{ultimo_turno}</b> |
            Longitud: <b>{est.longitud}</b> |
            Frutas comidas: <b>{f['frutas']}</b> |
            Recompensa acumulada: <b>{f['recompensa']:.1f}</b> |
            Estado: <b>{f['causa']}</b>
        </div>
        """
        with out_canvas:
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(5.8, 5.8))
            dibujar_tablero_snake(
                est,
                titulo=(
                    f"Turno {f['paso']} | Longitud: {est.longitud} | "
                    f"Frutas: {f['frutas']}\nFin: {f['causa']}"
                ),
                ax=ax,
            )
            plt.show()

    def iniciar_manual(_=None):
        """Inicia una partida manual con la instancia elegida."""
        memoria["modo"] = "manual"
        env = Snake(
            tamano=w_tam.value,
            semilla=w_sem.value,
            toroidal=w_toro.value,
            num_obstaculos=w_obs.value,
            num_frutas=w_frutas.value,
        )
        env.reset(semilla=w_sem.value)
        memoria["env_manual"] = env
        dibujar_manual(
            "Partida manual iniciada. Usa los botones Norte/Sur/Oeste/Este."
        )

    def paso_manual(direccion: str):
        """Juega ``direccion`` en la partida manual."""
        if memoria["modo"] != "manual" or memoria["env_manual"] is None:
            iniciar_manual()
        env: Snake = memoria["env_manual"]
        if not env.done:
            env.step(direccion)
        dibujar_manual(f"Acción jugada: {direccion.upper()}")

    def dibujar_manual(mensaje: str):
        """Dibuja la partida manual con ``mensaje`` en el recuadro."""
        env: Snake = memoria["env_manual"]
        recompensa_neta = env.ganancia_total - env.perdida_total
        info_html.value = f"""
        <div style="background:#fff3bf; padding:8px 12px;
                    border-radius:6px; font-family:monospace;">
            <b>MODO MANUAL (HUMANO)</b> | {mensaje} |
            Pasos: <b>{env.pasos}</b> |
            Longitud: <b>{env.estado.longitud}</b> |
            Frutas: <b>{env.frutas_comidas}</b> |
            Recompensa: <b>{recompensa_neta:.1f}</b> |
            Estado: <b>{env.causa_fin}</b>
        </div>
        """
        with out_canvas:
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(5.8, 5.8))
            dibujar_tablero_snake(
                env.estado,
                titulo=(
                    f"Modo Manual | Longitud: {env.estado.longitud} | "
                    f"Frutas: {env.frutas_comidas}\nFin: {env.causa_fin}"
                ),
                ax=ax,
            )
            plt.show()

    btn_simular.on_click(correr_simulacion_auto)
    btn_reiniciar_manual.on_click(iniciar_manual)
    slider_turno.observe(dibujar_frame_auto, names="value")
    btn_n.on_click(lambda _: paso_manual("norte"))
    btn_s.on_click(lambda _: paso_manual("sur"))
    btn_o.on_click(lambda _: paso_manual("oeste"))
    btn_e.on_click(lambda _: paso_manual("este"))

    correr_simulacion_auto()

    fila1 = widgets.HBox([w_tam, w_alg, w_heur, w_seg, w_toro])
    fila2 = widgets.HBox([w_sem, w_obs, w_frutas, w_pasos_sim])
    fila3 = widgets.HBox(
        [btn_simular, play_hist, slider_turno, btn_reiniciar_manual]
    )
    display(
        widgets.VBox([fila1, fila2, fila3, caja_manual, info_html, out_canvas])
    )
