"""Genera la tabla y la figura de resultados del informe.

Lee el CSV del benchmark de 100 semillas, resume cada configuración y
escribe ``tabla_resultados.tex``, ``fig_resultados.pdf`` y
``fig_resultados.png`` en la carpeta de salida.

Uso::

    python gen_resultados.py <ruta_csv> <carpeta_salida>
"""

import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Nombre de cada algoritmo en la tabla y en la figura.
NOMBRES = {
    "BFS": "BFS",
    "DFS": "DFS",
    "UCS": "UCS",
    "Avara": "Avara",
    "A*": "A*",
    "Beam": "Haz",
}

# Columnas que identifican una configuración del benchmark.
CLAVES = ["chequeo_seguridad", "algoritmo", "heuristica", "lambda_riesgo"]

# Paleta de la figura: azul, naranja, texto, texto secundario y rejilla.
AZ, NA, TX, TX2, GR = "#257de0", "#f47230", "#1f1f1f", "#6b6b6b", "#e6e6e6"

ENCABEZADO_TABLA = r"""\begin{table}[!htb]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{Resultados promedio sobre 100 semillas (0--99), tablero
$10\times10$, $k_{\max}=200$. Nodos y frontera son promedios por
decisión.}
\label{tab:resultados}
\begin{tabular}{lcccccrrrr}
\toprule
Algoritmo & $\lambda$ & Long. final & Vict. (\%) & Fin $k_{\max}$ (\%)
& Pasos/fruta & Gen. & Exp. & Front. máx. & $t$ (ms)\\
\midrule
"""

PIE_TABLA = r"""
\bottomrule
\end{tabular}
\end{table}
"""


def etiqueta(fila):
    """Devuelve el nombre de la configuración, p. ej. ``A* (Manh.)``."""
    alg = NOMBRES.get(fila.algoritmo, fila.algoritmo)
    abreviaturas = {"manhattan": " (Manh.)", "euclidea": " (Eucl.)"}
    return alg + abreviaturas.get(fila.heuristica, "")


def resumir(df):
    """Calcula las métricas promedio de cada configuración.

    Parameters
    ----------
    df : pandas.DataFrame
        Resultados crudos, una fila por partida.

    Returns
    -------
    pandas.DataFrame
        Una fila por configuración con sus promedios y su etiqueta.
    """
    df = df.copy()
    df["fin_kmax"] = df.causa_fin.str.startswith("Límite")
    df["corte"] = df.causa_fin.eq("En curso")

    res = (
        df.groupby(CLAVES, sort=False)
        .agg(
            Lm=("longitud_final", "mean"),
            Ls=("longitud_final", "std"),
            vic=("victoria", "mean"),
            kmax=("fin_kmax", "mean"),
            corte=("corte", "mean"),
            ppf=("pasos_por_fruta", "mean"),
            gen=("nodos_generados_prom", "mean"),
            exp=("nodos_expandidos_prom", "mean"),
            fro=("max_frontera_prom", "mean"),
            t=("tiempo_decision_ms", "mean"),
            n=("semilla", "count"),
        )
        .reset_index()
    )
    res["conf"] = res.apply(etiqueta, axis=1)
    return res


def escribir_tabla(res, ruta):
    """Escribe la tabla de resultados en formato LaTeX.

    Las configuraciones se agrupan en dos bloques: sin chequeo de
    seguridad y con chequeo de seguridad.
    """
    filas = []
    for seguridad in (False, True):
        if seguridad:
            filas.append(r"\midrule")
        titulo = (
            "Con chequeo de seguridad"
            if seguridad
            else "Sin chequeo de seguridad"
        )
        filas.append(r"\multicolumn{10}{l}{\textit{%s}}\\" % titulo)
        bloque = res[res.chequeo_seguridad == seguridad]
        for _, r in bloque.iterrows():
            filas.append(
                f"{r.conf} & {r.lambda_riesgo:.1f} & "
                f"{r.Lm:.1f} $\\pm$ {r.Ls:.1f} & "
                f"{100 * r.vic:.0f} & {100 * r.kmax:.0f} & "
                f"{r.ppf:.1f} & {r.gen:.1f} & {r.exp:.1f} & "
                f"{r.fro:.1f} & {r.t:.2f}\\\\"
            )
    tabla = ENCABEZADO_TABLA + "\n".join(filas) + PIE_TABLA
    with open(ruta, "w", encoding="utf-8") as archivo:
        archivo.write(tabla)


def graficar(res, carpeta):
    """Dibuja la longitud final y los nodos expandidos por algoritmo.

    Compara cada algoritmo sin y con chequeo de seguridad, usando
    lambda = 0 salvo en UCS, que usa lambda = 0.5. Guarda la figura en
    PDF y en PNG.
    """
    # Una configuración por algoritmo: se excluye A* con h_2.
    base = res[(res.lambda_riesgo == 0.0) | (res.algoritmo == "UCS")]
    base = base[~((base.algoritmo == "A*") & (base.heuristica == "euclidea"))]
    orden = ["BFS", "DFS", "UCS", "Avara", "A*", "Beam"]
    etiquetas = [NOMBRES[a] for a in orden]
    sin = base[~base.chequeo_seguridad].set_index("algoritmo").loc[orden]
    con = base[base.chequeo_seguridad].set_index("algoritmo").loc[orden]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.edgecolor": TX2,
            "axes.labelcolor": TX,
            "xtick.color": TX2,
            "ytick.color": TX2,
        }
    )
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.3))

    x = np.arange(len(orden))
    ancho = 0.38
    paneles = [
        ("Lm", "Longitud final media", 100),
        ("exp", "Nodos expandidos por decisión", None),
    ]
    for ax, (col, titulo, ymax) in zip(axs, paneles):
        ax.bar(
            x - ancho / 2 - 0.01,
            sin[col],
            ancho,
            color=AZ,
            label="Sin chequeo",
            zorder=3,
        )
        ax.bar(
            x + ancho / 2 + 0.01,
            con[col],
            ancho,
            color=NA,
            label="Con chequeo",
            zorder=3,
        )
        ax.set_xticks(x, etiquetas)
        ax.set_title(titulo, fontsize=8.5, color=TX, loc="left")
        ax.grid(axis="y", color=GR, lw=0.6, zorder=0)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        ax.tick_params(length=0)
        if ymax:
            # Línea punteada en 100: el tablero lleno.
            ax.set_ylim(0, ymax)
            ax.axhline(100, color=TX2, lw=0.6, ls=":")

    manejadores, nombres = axs[0].get_legend_handles_labels()
    fig.legend(
        manejadores,
        nombres,
        frameon=False,
        fontsize=7.5,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(f"{carpeta}/fig_resultados.pdf")
    fig.savefig(f"{carpeta}/fig_resultados.png", dpi=200)


def main():
    """Lee los argumentos, resume el CSV y genera tabla y figura."""
    ruta_csv, carpeta = sys.argv[1], sys.argv[2]
    res = resumir(pd.read_csv(ruta_csv))
    print(res.to_string())
    escribir_tabla(res, f"{carpeta}/tabla_resultados.tex")
    graficar(res, carpeta)


if __name__ == "__main__":
    main()
