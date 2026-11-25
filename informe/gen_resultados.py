import sys, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(csv)
df["fin_kmax"] = df.causa_fin.str.startswith("Límite")
df["corte"] = df.causa_fin.eq("En curso")

nombres = {"BFS":"BFS","DFS":"DFS","UCS":"UCS","Avara":"Avara","A*":"A*","Beam":"Haz"}
def etiqueta(r):
    alg = nombres.get(r.algoritmo, r.algoritmo)
    h = {"manhattan":" (Manh.)","euclidea":" (Eucl.)"}.get(r.heuristica,"")
    return alg + h

keys = ["chequeo_seguridad","algoritmo","heuristica","lambda_riesgo"]
g = df.groupby(keys, sort=False)
res = g.agg(Lm=("longitud_final","mean"), Ls=("longitud_final","std"),
            vic=("victoria","mean"), kmax=("fin_kmax","mean"), corte=("corte","mean"),
            ppf=("pasos_por_fruta","mean"), gen=("nodos_generados_prom","mean"),
            exp=("nodos_expandidos_prom","mean"), fro=("max_frontera_prom","mean"),
            t=("tiempo_decision_ms","mean"), n=("semilla","count")).reset_index()
res["conf"] = res.apply(etiqueta, axis=1)
print(res.to_string())

# ---------- tabla LaTeX ----------
filas = []
for seg in [False, True]:
    sub = res[res.chequeo_seguridad == seg]
    filas.append(r"\midrule" if seg else "")
    filas.append(r"\multicolumn{10}{l}{\textit{%s}}\\" % ("Con chequeo de seguridad" if seg else "Sin chequeo de seguridad"))
    for _, r in sub.iterrows():
        filas.append(f"{r.conf} & {r.lambda_riesgo:.1f} & {r.Lm:.1f} $\\pm$ {r.Ls:.1f} & {100*r.vic:.0f} & {100*r.kmax:.0f} & "
                     f"{r.ppf:.1f} & {r.gen:.1f} & {r.exp:.1f} & {r.fro:.1f} & {r.t:.2f}\\\\")
tabla = r"""\begin{table}[!htb]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{Resultados promedio sobre 100 semillas (0--99), tablero $10\times10$, $k_{\max}=200$. Nodos y frontera son promedios por decisión.}
\label{tab:resultados}
\begin{tabular}{lcccccrrrr}
\toprule
Algoritmo & $\lambda$ & Long. final & Vict. (\%) & Fin $k_{\max}$ (\%) & Pasos/fruta & Gen. & Exp. & Front. máx. & $t$ (ms)\\
""" + "\n".join(filas) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open(f"{out}/tabla_resultados.tex","w").write(tabla)

# ---------- figura ----------
base = res[(res.lambda_riesgo == 0.0) | (res.algoritmo == "UCS")]
base = base[~((base.algoritmo == "A*") & (base.heuristica == "euclidea"))]
orden = ["BFS","DFS","UCS","Avara","A*","Beam"]
lab = ["BFS","DFS","UCS","Avara","A*","Haz"]
sin = base[~base.chequeo_seguridad].set_index("algoritmo").loc[orden]
con = base[base.chequeo_seguridad].set_index("algoritmo").loc[orden]

AZ, NA, TX, TX2, GR = "#257de0", "#f47230", "#1f1f1f", "#6b6b6b", "#e6e6e6"
plt.rcParams.update({"font.family":"serif","font.size":8,"axes.edgecolor":TX2,
                     "axes.labelcolor":TX,"xtick.color":TX2,"ytick.color":TX2})
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.3))
import numpy as np
x = np.arange(len(orden)); w = 0.38
paneles = [("Lm","Longitud final media", 100), ("exp","Nodos expandidos por decisión", None)]
for ax, (col, titulo, ymax) in zip(axs, paneles):
    ax.bar(x - w/2 - 0.01, sin[col], w, color=AZ, label="Sin chequeo", zorder=3)
    ax.bar(x + w/2 + 0.01, con[col], w, color=NA, label="Con chequeo", zorder=3)
    ax.set_xticks(x, lab)
    ax.set_title(titulo, fontsize=8.5, color=TX, loc="left")
    ax.grid(axis="y", color=GR, lw=0.6, zorder=0)
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    if ymax: ax.set_ylim(0, ymax); ax.axhline(100, color=TX2, lw=0.6, ls=":")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=7.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(f"{out}/fig_resultados.pdf"); fig.savefig(f"{out}/fig_resultados.png", dpi=200)
