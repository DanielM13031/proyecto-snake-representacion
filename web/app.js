"use strict";
/* SnakeLab Studio: lógica de la página.
   El servidor (interfaz.py) corre los algoritmos de busqueda.py / agentes.py
   y devuelve trazas en JSON; aquí solo se dibujan y se animan. */

// =====================================================================
// Utilidades
// =====================================================================
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const clave = (c) => c[0] + "," + c[1];

async function api(ruta, datos) {
  const r = await fetch("api/" + ruta, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(datos || {}),
  });
  let j;
  try { j = await r.json(); } catch { j = { error: `HTTP ${r.status}` }; }
  if (!r.ok || j.error) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}

let temporizadorAviso = null;
function aviso(msg) {
  const el = $("#aviso");
  el.textContent = msg;
  el.classList.add("visible");
  clearTimeout(temporizadorAviso);
  temporizadorAviso = setTimeout(() => el.classList.remove("visible"), 6000);
}

function fmt(v, d = 0) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Number(v).toLocaleString("es-CO", { minimumFractionDigits: d, maximumFractionDigits: d });
}

async function conCarga(btn, fn) {
  const texto = btn.innerHTML;
  btn.disabled = true;
  btn.classList.add("cargando");
  try { return await fn(); }
  catch (e) { aviso("Error: " + e.message); console.error(e); }
  finally { btn.disabled = false; btn.classList.remove("cargando"); btn.innerHTML = texto; }
}

function chips(contenedor, atributo, alCambiar) {
  contenedor.addEventListener("click", (e) => {
    const b = e.target.closest(".chip");
    if (!b || !contenedor.contains(b)) return;
    $$(".chip", contenedor).forEach((c) => c.classList.toggle("activo", c === b));
    alCambiar(b.dataset[atributo], b);
  });
}
function activarChip(contenedor, atributo, valor) {
  $$(".chip", contenedor).forEach((c) => c.classList.toggle("activo", c.dataset[atributo] === valor));
}

function rango(id, formato = (v) => v) {
  const el = $("#" + id), out = $("#" + id + "-v");
  const act = () => { out.textContent = formato(el.value); };
  el.addEventListener("input", act);
  act();
  return el;
}

const NOMBRE_ALG = { BFS: "BFS", DFS: "DFS", UCS: "UCS", Avara: "Avara", "A*": "A*", Beam: "Haz" };
const NOMBRE_HEUR = { ninguna: "—", manhattan: "h₁ Manhattan", euclidea: "h₂ Euclídea", riesgo: "h₃ Manh. + riesgo" };
const INFORMADOS = new Set(["Avara", "A*", "Beam"]);
const DESC_ALG = {
  BFS: "No informada. Expande por niveles (cola FIFO). Encuentra el camino con menos pasos; es óptima si todos los pasos cuestan lo mismo (λ = 0).",
  DFS: "No informada. Se va lo más profundo que puede (pila LIFO). Gasta poca memoria, pero sus caminos suelen ser largos y no es óptima.",
  UCS: "No informada. Expande el nodo con menor costo acumulado g. Es óptima con costos positivos, también con λ > 0.",
  Avara: "Informada. Expande el nodo con menor h: va directo hacia lo que parece más cerca. Rápida, pero no garantiza el óptimo.",
  "A*": "Informada. Expande el nodo con menor f = g + h. Óptima si h no sobreestima (h₁ y h₂ cumplen; h₃ puede sobreestimar).",
  Beam: "Informada con memoria acotada: en cada nivel solo conserva los k mejores nodos por f. Puede no ser óptima o incluso fallar.",
};
const CRITERIO = {
  BFS: "sale de la frontera el nodo más antiguo (FIFO)",
  DFS: "sale de la frontera el nodo más reciente (LIFO)",
  UCS: "sale el nodo con menor g",
  Avara: "sale el nodo con menor h",
  "A*": "sale el nodo con menor f = g + h",
  Beam: "sale el mejor nodo por f entre los k que conserva el haz",
};
const DIRS = { norte: [0, 1], sur: [0, -1], este: [1, 0], oeste: [-1, 0] };

function direccionEntre(a, b, N) {
  let dx = b[0] - a[0], dy = b[1] - a[1];
  if (Math.abs(dx) > 1) dx = -Math.sign(dx);
  if (Math.abs(dy) > 1) dy = -Math.sign(dy);
  for (const [nom, d] of Object.entries(DIRS)) if (d[0] === dx && d[1] === dy) return nom;
  return null;
}
function adyacentes(a, b, N, toroidal) {
  const dx = Math.abs(a[0] - b[0]), dy = Math.abs(a[1] - b[1]);
  if (dx + dy === 1) return true;
  if (!toroidal) return false;
  return (dx === N - 1 && dy === 0) || (dy === N - 1 && dx === 0);
}

// =====================================================================
// Dibujo del tablero
// =====================================================================
const COLORES = {
  fondoA: "#0d1526", fondoB: "#101b30", linea: "rgba(148,163,184,0.07)",
  expandido: "rgba(59,130,246,0.40)", frontera: "rgba(242,166,0,0.60)", bordeFrontera: "rgba(253,224,71,0.9)",
  actual: "#f47230", camino: "#fb923c", solucion: "#2dd4bf", plan: "#a3e635", evasion: "#ee7fac",
  muro: "#334155", muroLuz: "#4b5d7a", cabeza: "#22c55e", cola: [134, 239, 172], cuerpo: [22, 163, 74],
};

class Tablero {
  constructor(canvas, opciones = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.op = Object.assign({ maxLado: 680, margenAlto: 300, fijo: null }, opciones);
    this.escena = null;
    this.lado = 0;
    if (this.op.fijo) {
      this.fijar(this.op.fijo);
    } else {
      const obs = new ResizeObserver(() => this.ajustar());
      obs.observe(canvas.parentElement);
      window.addEventListener("resize", () => this.ajustar());
    }
  }
  fijar(lado) {
    const dpr = window.devicePixelRatio || 1;
    this.lado = lado;
    this.canvas.width = Math.round(lado * dpr);
    this.canvas.height = Math.round(lado * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  ajustar() {
    const ancho = this.canvas.parentElement.clientWidth;
    if (!ancho) return;
    const alto = this.op.margenAlto ? Math.max(260, window.innerHeight - this.op.margenAlto) : Infinity;
    const lado = Math.floor(Math.min(ancho, this.op.maxLado, alto));
    if (lado === this.lado) return;
    this.canvas.style.width = lado + "px";
    this.canvas.style.height = lado + "px";
    this.fijar(lado);
    if (this.escena) this.dibujar(this.escena);
  }
  celdaEn(evento) {
    if (!this.escena) return null;
    const r = this.canvas.getBoundingClientRect();
    const N = this.escena.N;
    const x = Math.floor(((evento.clientX - r.left) / r.width) * N);
    const yv = Math.floor(((evento.clientY - r.top) / r.height) * N);
    if (x < 0 || yv < 0 || x >= N || yv >= N) return null;
    return [x, N - 1 - yv];
  }

  dibujar(esc) {
    this.escena = esc;
    if (!this.lado) { this.ajustar(); if (!this.lado) return; }
    const ctx = this.ctx, L = this.lado, N = esc.N, s = L / N;
    const X = (c) => (c[0] + 0.5) * s, Y = (c) => (N - 1 - c[1] + 0.5) * s;
    const caja = (c, inset = 0) => [c[0] * s + inset, (N - 1 - c[1]) * s + inset, s - 2 * inset, s - 2 * inset];
    ctx.clearRect(0, 0, L, L);

    // fondo cuadriculado
    for (let x = 0; x < N; x++) for (let y = 0; y < N; y++) {
      ctx.fillStyle = (x + y) % 2 ? COLORES.fondoB : COLORES.fondoA;
      ctx.fillRect(x * s, y * s, s, s);
    }
    ctx.strokeStyle = COLORES.linea; ctx.lineWidth = 1;
    for (let i = 1; i < N; i++) {
      ctx.beginPath(); ctx.moveTo(i * s, 0); ctx.lineTo(i * s, L); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i * s); ctx.lineTo(L, i * s); ctx.stroke();
    }

    const tenue = esc.tenue ? 0.45 : 1;
    ctx.globalAlpha = tenue;
    if (esc.expandidos) {
      ctx.fillStyle = COLORES.expandido;
      for (const c of esc.expandidos) { const b = caja(c, 1.5); redondeado(ctx, ...b, s * 0.14); ctx.fill(); }
    }
    if (esc.frontera) {
      for (const c of esc.frontera) {
        const b = caja(c, 2);
        ctx.fillStyle = COLORES.frontera; redondeado(ctx, ...b, s * 0.14); ctx.fill();
        ctx.strokeStyle = COLORES.bordeFrontera; ctx.lineWidth = 1.2; ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;

    // muros
    for (const c of esc.muros || []) {
      const b = caja(c, 1.5);
      ctx.fillStyle = COLORES.muro; redondeado(ctx, ...b, s * 0.12); ctx.fill();
      ctx.fillStyle = COLORES.muroLuz; redondeado(ctx, b[0], b[1], b[2], b[3] * 0.32, s * 0.12); ctx.fill();
    }

    // marco del modo toroidal
    if (esc.toroidal) {
      ctx.save(); ctx.setLineDash([6, 6]); ctx.strokeStyle = "rgba(45,212,191,0.65)"; ctx.lineWidth = 2;
      ctx.strokeRect(1, 1, L - 2, L - 2); ctx.restore();
    }

    if (esc.caminoParcial) linea(ctx, esc.caminoParcial, X, Y, N, esc.toroidal, COLORES.camino, Math.max(2, s * 0.09), null, 0.95);
    if (esc.solucion) linea(ctx, esc.solucion, X, Y, N, esc.toroidal, COLORES.solucion, Math.max(3, s * 0.13), null, esc.solucionTenue ? 0.4 : 0.95);
    if (esc.plan && esc.plan.length) {
      const desde = esc.cuerpo && esc.cuerpo.length ? [esc.cuerpo[0], ...esc.plan] : esc.plan;
      linea(ctx, desde, X, Y, N, esc.toroidal, esc.planColor || COLORES.plan, Math.max(2.5, s * 0.1), [s * 0.22, s * 0.16], 0.95);
    }
    if (esc.sugerencia) linea(ctx, esc.sugerencia, X, Y, N, esc.toroidal, COLORES.solucion, Math.max(3, s * 0.12), [s * 0.25, s * 0.15], 0.95);

    if (esc.actual) {
      const b = caja(esc.actual, 2.5);
      ctx.save(); ctx.shadowColor = COLORES.actual; ctx.shadowBlur = 12;
      ctx.strokeStyle = COLORES.actual; ctx.lineWidth = 3; redondeado(ctx, ...b, s * 0.16); ctx.stroke(); ctx.restore();
    }

    for (const f of esc.frutas || []) dibujarFruta(ctx, X(f), Y(f), s);
    if (esc.cuerpo && esc.cuerpo.length) dibujarSerpiente(ctx, esc, X, Y, s);
    if (esc.destello && esc.cuerpo) {
      const h = esc.cuerpo[0];
      ctx.save(); ctx.strokeStyle = "rgba(253,224,71,0.9)"; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(X(h), Y(h), s * 0.62, 0, Math.PI * 2); ctx.stroke(); ctx.restore();
    }
    if (esc.hover) {
      const b = caja(esc.hover.c, 1);
      ctx.save(); ctx.setLineDash([4, 3]); ctx.strokeStyle = esc.hover.color; ctx.lineWidth = 2;
      redondeado(ctx, ...b, s * 0.14); ctx.stroke(); ctx.restore();
    }
    if (esc.mensaje) dibujarMensaje(ctx, L, esc.mensaje);
  }
}

function redondeado(ctx, x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function linea(ctx, celdas, X, Y, N, toroidal, color, ancho, guiones, alfa) {
  if (!celdas || celdas.length < 2) return;
  ctx.save();
  ctx.globalAlpha = alfa;
  ctx.strokeStyle = color; ctx.lineWidth = ancho; ctx.lineCap = "round"; ctx.lineJoin = "round";
  if (guiones) ctx.setLineDash(guiones);
  ctx.beginPath();
  ctx.moveTo(X(celdas[0]), Y(celdas[0]));
  for (let i = 1; i < celdas.length; i++) {
    const a = celdas[i - 1], b = celdas[i];
    if (Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) === 1) ctx.lineTo(X(b), Y(b));
    else ctx.moveTo(X(b), Y(b)); // salto por el borde toroidal
  }
  ctx.stroke();
  const u = celdas[celdas.length - 1];
  ctx.setLineDash([]); ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(X(u), Y(u), ancho * 0.9, 0, Math.PI * 2); ctx.fill();
  ctx.restore();
}

function dibujarFruta(ctx, x, y, s) {
  const r = s * 0.3;
  const g = ctx.createRadialGradient(x - r * 0.35, y - r * 0.35, r * 0.15, x, y, r);
  g.addColorStop(0, "#fca5a5"); g.addColorStop(0.35, "#ef4444"); g.addColorStop(1, "#991b1b");
  ctx.save();
  ctx.shadowColor = "rgba(239,68,68,0.55)"; ctx.shadowBlur = s * 0.25;
  ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y + r * 0.08, r, 0, Math.PI * 2); ctx.fill();
  ctx.restore();
  ctx.strokeStyle = "#78350f"; ctx.lineWidth = Math.max(1.2, s * 0.05);
  ctx.beginPath(); ctx.moveTo(x, y - r * 0.75); ctx.lineTo(x + r * 0.15, y - r * 1.2); ctx.stroke();
  ctx.fillStyle = "#4ade80";
  ctx.beginPath(); ctx.ellipse(x + r * 0.45, y - r * 1.05, r * 0.38, r * 0.18, -0.5, 0, Math.PI * 2); ctx.fill();
}

function mezclar(a, b, t) { return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(",")})`; }

function dibujarSerpiente(ctx, esc, X, Y, s) {
  const c = esc.cuerpo, n = c.length;
  const ancho = s * 0.66;
  ctx.save();
  ctx.lineCap = "round";
  for (let i = n - 1; i >= 1; i--) {
    const t = n > 1 ? (i / (n - 1)) : 0;
    const color = mezclar(COLORES.cuerpo, COLORES.cola, t);
    const a = c[i], b = c[i - 1];
    ctx.strokeStyle = color; ctx.fillStyle = color;
    ctx.lineWidth = i === n - 1 ? ancho * 0.72 : ancho;
    if (Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) === 1) {
      ctx.beginPath(); ctx.moveTo(X(a), Y(a)); ctx.lineTo(X(b), Y(b)); ctx.stroke();
    } else {
      ctx.beginPath(); ctx.arc(X(a), Y(a), ctx.lineWidth / 2, 0, Math.PI * 2); ctx.fill();
    }
  }
  // cabeza
  const h = c[0];
  const hx = X(h), hy = Y(h);
  ctx.fillStyle = COLORES.cabeza;
  ctx.shadowColor = "rgba(34,197,94,0.6)"; ctx.shadowBlur = s * 0.3;
  ctx.beginPath(); ctx.arc(hx, hy, s * 0.4, 0, Math.PI * 2); ctx.fill();
  ctx.shadowBlur = 0;
  // ojos hacia donde mira
  let dir = esc.dirCabeza || "norte";
  if (n >= 2) dir = direccionEntre(c[1], h, esc.N) || dir;
  const d = DIRS[dir];
  const ux = d[0], uy = -d[1];          // en pantalla y crece hacia abajo
  const px = -uy, py = ux;
  for (const lado of [-1, 1]) {
    const ex = hx + ux * s * 0.12 + px * s * 0.17 * lado;
    const ey = hy + uy * s * 0.12 + py * s * 0.17 * lado;
    ctx.fillStyle = "#f8fafc"; ctx.beginPath(); ctx.arc(ex, ey, s * 0.1, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#0f172a"; ctx.beginPath(); ctx.arc(ex + ux * s * 0.04, ey + uy * s * 0.04, s * 0.05, 0, Math.PI * 2); ctx.fill();
  }
  ctx.restore();
}

function dibujarMensaje(ctx, L, m) {
  ctx.save();
  ctx.fillStyle = "rgba(6,10,20,0.55)"; ctx.fillRect(0, 0, L, L);
  const w = Math.min(L * 0.86, 420), h = m.sub ? 86 : 58;
  const x = (L - w) / 2, y = (L - h) / 2;
  ctx.fillStyle = "rgba(17,26,44,0.96)";
  ctx.strokeStyle = m.tono === "mal" ? "#ef4444" : m.tono === "neutro" ? "#f2a600" : "#22c55e";
  ctx.lineWidth = 2; redondeado(ctx, x, y, w, h, 14); ctx.fill(); ctx.stroke();
  ctx.fillStyle = "#e6edf7"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.font = "800 18px Inter, Segoe UI, system-ui, sans-serif";
  ctx.fillText(m.titulo, L / 2, y + (m.sub ? 30 : h / 2));
  if (m.sub) {
    ctx.fillStyle = "#93a4bf"; ctx.font = "600 13px Inter, Segoe UI, system-ui, sans-serif";
    envolver(ctx, m.sub, L / 2, y + 58, w - 28, 16);
  }
  ctx.restore();
}
function envolver(ctx, texto, x, y, ancho, alto) {
  const palabras = texto.split(" ");
  let lineaAct = "", lineas = [];
  for (const p of palabras) {
    const prueba = lineaAct ? lineaAct + " " + p : p;
    if (ctx.measureText(prueba).width > ancho && lineaAct) { lineas.push(lineaAct); lineaAct = p; }
    else lineaAct = prueba;
  }
  lineas.push(lineaAct);
  lineas.slice(0, 2).forEach((l, i) => ctx.fillText(l, x, y + i * alto - ((Math.min(lineas.length, 2) - 1) * alto) / 2));
}

// =====================================================================
// Reproductor (play / pausa / paso a paso / retroceso / velocidad)
// =====================================================================
const ICONOS = {
  inicio: '<svg viewBox="0 0 24 24"><path d="M6 5h2v14H6zM20 5v14L9 12z"/></svg>',
  atras: '<svg viewBox="0 0 24 24"><path d="M17 5v14L6 12z"/></svg>',
  play: '<svg viewBox="0 0 24 24"><path d="M7 4v16l13-8z"/></svg>',
  pausa: '<svg viewBox="0 0 24 24"><path d="M6 4h4v16H6zM14 4h4v16h-4z"/></svg>',
  adelante: '<svg viewBox="0 0 24 24"><path d="M7 5v14l11-7z"/></svg>',
  fin: '<svg viewBox="0 0 24 24"><path d="M16 5h2v14h-2zM4 5v14l11-7z"/></svg>',
};

class Reproductor {
  constructor(contenedor, { fps = 4, alCambiar, texto }) {
    this.el = contenedor;
    this.fps = fps;
    this.vel = 1;
    this.alCambiar = alCambiar;
    this.texto = texto || ((i, n) => `Paso ${i + 1} / ${n}`);
    this.total = 0; this.i = 0; this.corriendo = false; this.acum = 0; this.previo = 0;
    contenedor.innerHTML = `
      <button class="btn" data-a="inicio" title="Ir al inicio">${ICONOS.inicio}</button>
      <button class="btn" data-a="atras" title="Paso atrás (←)">${ICONOS.atras}</button>
      <button class="btn play" data-a="play" title="Reproducir / pausar (espacio)">${ICONOS.play}</button>
      <button class="btn" data-a="adelante" title="Paso adelante (→)">${ICONOS.adelante}</button>
      <button class="btn" data-a="fin" title="Ir al final">${ICONOS.fin}</button>
      <div class="linea"><input type="range" min="0" max="0" value="0"></div>
      <span class="contador">—</span>
      <select title="Velocidad">
        ${[0.25, 0.5, 1, 2, 4, 8, 16, 32].map((v) => `<option value="${v}" ${v === 1 ? "selected" : ""}>${v}×</option>`).join("")}
      </select>`;
    this.rango = $("input", contenedor);
    this.contador = $(".contador", contenedor);
    this.btnPlay = $('[data-a="play"]', contenedor);
    contenedor.addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      const a = b.dataset.a;
      if (a === "play") this.alternar();
      else { this.pausar(); if (a === "inicio") this.ir(0); if (a === "fin") this.ir(this.total - 1); if (a === "atras") this.ir(this.i - 1); if (a === "adelante") this.ir(this.i + 1); }
    });
    this.rango.addEventListener("input", () => { this.pausar(); this.ir(+this.rango.value); });
    $("select", contenedor).addEventListener("change", (e) => { this.vel = +e.target.value; });
    this.el.classList.add("inactivo");
  }
  cargar(total, { autoplay = true } = {}) {
    this.total = total; this.rango.max = Math.max(0, total - 1);
    this.el.classList.toggle("inactivo", total <= 1);
    this.ir(0);
    if (autoplay && total > 1) this.reproducir(); else this.pausar();
  }
  vaciar() { this.pausar(); this.total = 0; this.rango.max = 0; this.contador.textContent = "—"; this.el.classList.add("inactivo"); }
  ir(i) {
    if (!this.total) return;
    this.i = Math.max(0, Math.min(this.total - 1, i));
    this.rango.value = this.i;
    this.contador.textContent = this.texto(this.i, this.total);
    this.alCambiar(this.i);
  }
  reproducir() {
    if (this.i >= this.total - 1) this.ir(0);
    this.corriendo = true; this.btnPlay.innerHTML = ICONOS.pausa;
    this.previo = performance.now(); this.acum = 0;
    const paso = (t) => {
      if (!this.corriendo) return;
      this.acum += (t - this.previo) * this.fps * this.vel; this.previo = t;
      if (this.acum >= 1000) {
        const n = Math.floor(this.acum / 1000); this.acum -= n * 1000;
        this.ir(this.i + n);
        if (this.i >= this.total - 1) { this.pausar(); return; }
      }
      requestAnimationFrame(paso);
    };
    requestAnimationFrame(paso);
  }
  pausar() { this.corriendo = false; this.btnPlay.innerHTML = ICONOS.play; }
  alternar() { this.corriendo ? this.pausar() : this.reproducir(); }
}

// =====================================================================
// Gráficos SVG
// =====================================================================
const SVGNS = "http://www.w3.org/2000/svg";
function svgEl(tag, atrs = {}, texto) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(atrs)) e.setAttribute(k, v);
  if (texto !== undefined) e.textContent = texto;
  return e;
}
function prepararSvg(svg, altoPorDefecto) {
  const w = Math.max(svg.clientWidth, svg.parentElement.clientWidth - 32);
  if (w < 120) return [0, 0]; // pestaña oculta o en plena transición: no dibujar
  const h = svg.clientHeight || altoPorDefecto;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.innerHTML = "";
  return [w, h];
}
function escalaBonita(max) {
  if (max <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(max)));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= max) return m * p;
  return 10 * p;
}

function graficoLineas(svg, series, { ymax, cursor, alto = 150, refY, etiquetaRef, area = false } = {}) {
  const [W, H] = prepararSvg(svg, alto);
  if (!W) return;
  const m = { l: 34, r: 10, t: 10, b: 18 };
  const xmax = Math.max(1, ...series.map((s) => s.valores.length - 1));
  const ytop = ymax || escalaBonita(Math.max(1, ...series.flatMap((s) => s.valores)));
  const x = (i) => m.l + (i / xmax) * (W - m.l - m.r);
  const y = (v) => H - m.b - (v / ytop) * (H - m.t - m.b);
  for (const f of [0, 0.5, 1]) {
    const v = ytop * f;
    svg.append(svgEl("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: f === 0 ? "eje" : "rejilla" }));
    svg.append(svgEl("text", { x: m.l - 6, y: y(v) + 4, "text-anchor": "end" }, fmt(v, Number.isInteger(v) ? 0 : 1)));
  }
  if (refY !== undefined) {
    svg.append(svgEl("line", { x1: m.l, x2: W - m.r, y1: y(refY), y2: y(refY), stroke: "#64748b", "stroke-dasharray": "3 4" }));
    if (etiquetaRef) svg.append(svgEl("text", { x: W - m.r, y: y(refY) - 4, "text-anchor": "end" }, etiquetaRef));
  }
  for (const s of series) {
    if (!s.valores.length) continue;
    const pts = s.valores.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
    if (area) svg.append(svgEl("polygon", { points: `${x(0)},${y(0)} ${pts} ${x(s.valores.length - 1)},${y(0)}`, fill: s.color, opacity: 0.15 }));
    svg.append(svgEl("polyline", { points: pts, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round" }));
  }
  if (cursor !== undefined) {
    svg.append(svgEl("line", { x1: x(cursor), x2: x(cursor), y1: m.t, y2: H - m.b, stroke: "#e6edf7", "stroke-width": 1, opacity: 0.5 }));
    for (const s of series) {
      const i = Math.min(cursor, s.valores.length - 1);
      if (i >= 0) svg.append(svgEl("circle", { cx: x(i), cy: y(s.valores[i]), r: 4, fill: s.color, stroke: "#0a0f1c", "stroke-width": 2 }));
    }
  }
}

function graficoBarrasH(svg, items, { formato = (v) => fmt(v), alto } = {}) {
  const filas = items.length;
  const [W, H] = prepararSvg(svg, alto || filas * 26 + 16);
  if (!W) return;
  svg.setAttribute("viewBox", `0 0 ${W} ${filas * 26 + 16}`);
  svg.style.height = (filas * 26 + 16) + "px";
  const m = { l: 150, r: 70 };
  const max = Math.max(1e-9, ...items.map((d) => d.valor || 0));
  items.forEach((d, i) => {
    const y = 8 + i * 26;
    svg.append(svgEl("text", { x: m.l - 10, y: y + 13, "text-anchor": "end" }, d.etiqueta));
    if (d.valor === null || d.valor === undefined) {
      svg.append(svgEl("text", { x: m.l, y: y + 13, fill: "#fca5a5" }, "falló"));
      return;
    }
    const w = Math.max(2, ((W - m.l - m.r) * d.valor) / max);
    svg.append(svgEl("rect", { x: m.l, y: y + 3, width: w, height: 16, rx: 4, fill: d.color }));
    svg.append(svgEl("text", { x: m.l + w + 6, y: y + 15, class: "valor" }, formato(d.valor)));
  });
}

function graficoBarrasAgrupadas(svg, categorias, series, { formato = (v) => fmt(v), ymax } = {}) {
  const [W, H] = prepararSvg(svg, 300);
  if (!W) return;
  const m = { l: 44, r: 10, t: 22, b: 28 };
  const top = ymax || escalaBonita(Math.max(1, ...series.flatMap((s) => s.valores)));
  const y = (v) => H - m.b - (v / top) * (H - m.t - m.b);
  for (const f of [0, 0.25, 0.5, 0.75, 1]) {
    svg.append(svgEl("line", { x1: m.l, x2: W - m.r, y1: y(top * f), y2: y(top * f), class: f === 0 ? "eje" : "rejilla" }));
    svg.append(svgEl("text", { x: m.l - 6, y: y(top * f) + 4, "text-anchor": "end" }, fmt(top * f)));
  }
  const ancho = (W - m.l - m.r) / categorias.length;
  const bw = Math.min(34, (ancho * 0.7) / series.length);
  categorias.forEach((cat, ci) => {
    const x0 = m.l + ci * ancho + (ancho - bw * series.length - 2 * (series.length - 1)) / 2;
    series.forEach((s, si) => {
      const v = s.valores[ci];
      if (v === null || v === undefined) return;
      const x = x0 + si * (bw + 2);
      const h = Math.max(1, H - m.b - y(v));
      svg.append(svgEl("rect", { x, y: y(v), width: bw, height: h, rx: 4, fill: s.color }));
      svg.append(svgEl("text", { x: x + bw / 2, y: y(v) - 5, "text-anchor": "middle", class: "valor" }, formato(v)));
    });
    svg.append(svgEl("text", { x: m.l + ci * ancho + ancho / 2, y: H - 8, "text-anchor": "middle" }, cat));
  });
}

// =====================================================================
// Navegación entre pestañas
// =====================================================================
const VISTAS = {};
let vistaActual = "busqueda";
$("#pestanas").addEventListener("click", (e) => {
  const b = e.target.closest(".pestana"); if (!b) return;
  irAVista(b.dataset.vista);
});
// quita el foco de los botones para que la barra espaciadora no los vuelva a activar
document.addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) b.blur(); });

function irAVista(nombre) {
  vistaActual = nombre;
  $$(".pestana").forEach((p) => p.classList.toggle("activa", p.dataset.vista === nombre));
  $$(".vista").forEach((v) => v.classList.toggle("activa", v.id === "vista-" + nombre));
  requestAnimationFrame(() => VISTAS[nombre]?.alMostrar?.());
}

document.addEventListener("keydown", (e) => {
  if (e.target.closest("input, select, textarea")) return;
  const v = VISTAS[vistaActual];
  if (v?.teclado) v.teclado(e);
});

// =====================================================================
// 1. BÚSQUEDA PASO A PASO
// =====================================================================
(function vistaBusqueda() {
  const B = { inst: null, alg: "A*", herr: null, datos: null, arrastre: null, hover: null, nPasos: 0 };
  const tablero = new Tablero($("#b-canvas"), { maxLado: 680, margenAlto: 330 });
  const PRESETS = { facil: { L: 1, M: 0 }, media: { L: 12, M: 6 }, dificil: { L: 32, M: 14 } };

  rango("b-N"); rango("b-L"); rango("b-M");
  rango("b-lam", (v) => Number(v).toFixed(1)); rango("b-k");

  chips($("#b-dificultad"), "dif", (d) => {
    $("#b-L").value = PRESETS[d].L; $("#b-M").value = PRESETS[d].M;
    $("#b-L").dispatchEvent(new Event("input")); $("#b-M").dispatchEvent(new Event("input"));
    generar();
  });
  chips($("#b-algoritmos"), "alg", (a) => { B.alg = a; actualizarControlesAlg(); });
  $("#b-heur").addEventListener("change", actualizarControlesAlg);
  $("#b-dado").addEventListener("click", () => { $("#b-semilla").value = Math.floor(Math.random() * 10000); generar(); });
  $("#b-generar").addEventListener("click", generar);
  $("#b-resolver").addEventListener("click", () => conCarga($("#b-resolver"), resolver));
  ["b-N", "b-L", "b-M"].forEach((id) => $("#" + id).addEventListener("change", () => { activarChip($("#b-dificultad"), "dif", ""); generar(); }));
  $("#b-toro").addEventListener("change", generar);

  const AYUDA_HERR = {
    fruta: "Clic en una celda libre para mover la fruta.",
    muro: "Clic o arrastra para poner o quitar muros.",
    serpiente: "Presiona donde va la cabeza y arrastra hacia la cola por celdas vecinas. Devuélvete para borrar.",
  };
  $("#b-herramientas").addEventListener("click", (e) => {
    const b = e.target.closest(".chip"); if (!b) return;
    B.herr = B.herr === b.dataset.herr ? null : b.dataset.herr;
    $$("#b-herramientas .chip").forEach((c) => c.classList.toggle("activo", c.dataset.herr === B.herr));
    $("#b-ayuda-herr").textContent = B.herr ? AYUDA_HERR[B.herr] : "Elige una herramienta y haz clic en el tablero.";
    $("#b-canvas").style.cursor = B.herr ? "crosshair" : "default";
  });

  const repro = new Reproductor($("#b-repro"), {
    fps: 4,
    alCambiar: (i) => mostrarPaso(i),
    texto: (i, n) => (i < B.nPasos ? `Iteración ${i + 1} / ${B.nPasos}` : `Movimiento ${i - B.nPasos + 1} / ${n - B.nPasos}`),
  });

  function actualizarControlesAlg() {
    const inf = INFORMADOS.has(B.alg);
    $("#b-heur-ctl").style.display = inf ? "" : "none";
    $("#b-k-ctl").style.display = B.alg === "Beam" ? "" : "none";
    let d = DESC_ALG[B.alg];
    if (inf && $("#b-heur").value === "riesgo") d += " Con h₃ la heurística puede sobreestimar el costo real.";
    $("#b-desc-alg").textContent = d;
  }
  actualizarControlesAlg();

  async function generar() {
    try {
      const r = await api("instancia", {
        N: +$("#b-N").value, longitud: +$("#b-L").value, muros: +$("#b-M").value,
        semilla: +$("#b-semilla").value, toroidal: $("#b-toro").checked,
      });
      B.inst = r.estado;
      limpiarResultado();
    } catch (e) { aviso("No se pudo generar: " + e.message); }
  }

  function limpiarResultado() {
    B.datos = null; B.nPasos = 0;
    repro.vaciar();
    $("#b-fase").className = "insignia"; $("#b-fase").textContent = "Instancia lista";
    $("#b-resultado").innerHTML = '<p class="ayuda">Aún no hay resultado.</p>';
    $("#b-estado").innerHTML = '<p class="ayuda" style="grid-column:1/-1;margin:0">Aparece al resolver: nodo actual, g, h, f y tamaño de la frontera en cada iteración.</p>';
    $("#b-sparkline").innerHTML = "";
    $("#b-explicacion").innerHTML = "Presiona <b>Resolver</b> para ver cómo explora el algoritmo.";
    infoInstancia();
    dibujarInstancia();
  }

  function infoInstancia() {
    const e = B.inst; if (!e) return;
    const f = e.fruta ? `fruta en (${e.fruta[0]}, ${e.fruta[1]})` : "sin fruta";
    $("#b-info-inst").textContent = `${e.N}×${e.N} · L = ${e.cuerpo.length} · ${e.muros.length} muros · ${f}${e.toroidal ? " · toroidal" : ""}`;
  }

  function escenaBase() {
    const e = B.inst;
    return { N: e.N, cuerpo: e.cuerpo, frutas: e.fruta ? [e.fruta] : [], muros: e.muros, toroidal: e.toroidal };
  }
  function dibujarInstancia() {
    if (!B.inst) return;
    const esc = escenaBase();
    if (B.hover && B.herr) esc.hover = { c: B.hover, color: { fruta: "#ef4444", muro: "#94a3b8", serpiente: "#22c55e" }[B.herr] };
    tablero.dibujar(esc);
  }

  // ---------- edición con el mouse ----------
  const lienzo = $("#b-canvas");
  const ocupada = (c, sinCuerpo = false) => {
    const k = clave(c), e = B.inst;
    return (!sinCuerpo && e.cuerpo.some((x) => clave(x) === k)) || e.muros.some((x) => clave(x) === k) || (e.fruta && clave(e.fruta) === k);
  };
  lienzo.addEventListener("pointerdown", (ev) => {
    if (!B.herr || !B.inst) return;
    const c = tablero.celdaEn(ev); if (!c) return;
    lienzo.setPointerCapture(ev.pointerId);
    const e = B.inst, k = clave(c);
    if (B.herr === "fruta") {
      if (!e.cuerpo.some((x) => clave(x) === k) && !e.muros.some((x) => clave(x) === k)) e.fruta = c;
    } else if (B.herr === "muro") {
      const existe = e.muros.some((x) => clave(x) === k);
      B.arrastre = { modo: existe ? "quitar" : "poner" };
      pintarMuro(c);
    } else if (B.herr === "serpiente") {
      if (e.muros.some((x) => clave(x) === k) || (e.fruta && clave(e.fruta) === k)) return;
      e.cuerpo = [c];
      B.arrastre = { modo: "serpiente" };
    }
    editado();
  });
  lienzo.addEventListener("pointermove", (ev) => {
    if (!B.inst) return;
    const c = tablero.celdaEn(ev);
    const cambio = (c && clave(c)) !== (B.hover && clave(B.hover));
    B.hover = c;
    if (B.arrastre && c) {
      const e = B.inst;
      if (B.arrastre.modo === "serpiente") {
        const ult = e.cuerpo[e.cuerpo.length - 1];
        const pen = e.cuerpo[e.cuerpo.length - 2];
        if (pen && clave(pen) === clave(c)) { e.cuerpo.pop(); editado(); return; }
        if (adyacentes(ult, c, e.N, e.toroidal) && !ocupada(c)) { e.cuerpo.push(c); editado(); return; }
      } else pintarMuro(c);
    }
    if (cambio && !B.datos) dibujarInstancia();
  });
  const soltar = () => { B.arrastre = null; };
  lienzo.addEventListener("pointerup", soltar);
  lienzo.addEventListener("pointercancel", soltar);
  lienzo.addEventListener("pointerleave", () => { B.hover = null; if (!B.datos) dibujarInstancia(); });

  function pintarMuro(c) {
    const e = B.inst, k = clave(c);
    if (e.cuerpo.some((x) => clave(x) === k) || (e.fruta && clave(e.fruta) === k)) return;
    const i = e.muros.findIndex((x) => clave(x) === k);
    if (B.arrastre.modo === "poner" && i < 0) e.muros.push(c);
    if (B.arrastre.modo === "quitar" && i >= 0) e.muros.splice(i, 1);
    editado();
  }
  function editado() {
    activarChip($("#b-dificultad"), "dif", "");
    if (B.datos) limpiarResultado(); else { infoInstancia(); dibujarInstancia(); }
  }

  // ---------- resolver ----------
  async function resolver() {
    const e = B.inst;
    if (!e || !e.fruta) { aviso("Pon una fruta en el tablero (herramienta 🍎)."); return; }
    const heur = INFORMADOS.has(B.alg) ? $("#b-heur").value : "ninguna";
    const r = await api("buscar", {
      estado: e, algoritmo: B.alg, heuristica: heur, lambda: +$("#b-lam").value, k: +$("#b-k").value,
    });
    B.datos = r; B.datos.heur = heur; B.datos.alg = B.alg;
    B.nPasos = r.pasos.length;
    const movimientos = r.resultado.exito ? r.cuerpos.length - 1 : 0;
    mostrarResultado();
    repro.cargar(B.nPasos + movimientos);
  }

  function mostrarResultado() {
    const R = B.datos.resultado;
    const kpi = (t, v, cls = "") => `<div class="kpi ${cls}"><div class="t">${t}</div><div class="v">${v}</div></div>`;
    $("#b-resultado").innerHTML = [
      kpi("¿Llegó a la fruta?", R.exito ? "Sí" : "No", R.exito ? "bien" : "mal"),
      kpi("Tiempo", `${fmt(R.tiempo_ms, 2)} <small>ms</small>`),
      kpi("Longitud", R.exito ? `${R.longitud} <small>pasos</small>` : "—"),
      kpi("Costo", R.exito ? fmt(R.costo, 2) : "—"),
      kpi("Generados", fmt(R.generados)),
      kpi("Expandidos", fmt(R.expandidos)),
      kpi("Frontera máx.", fmt(R.max_frontera)),
      kpi("Óptimo garantizado", R.optimo_teorico ? "Sí" : "No", R.optimo_teorico ? "bien" : ""),
      R.exito ? kpi("Tras comer, ¿hay camino a la cola?", R.seguro ? "Sí: estado seguro" : "No: podría encerrarse", "ancho " + (R.seguro ? "bien" : "mal")) : "",
    ].join("");
  }

  function mostrarPaso(i) {
    const D = B.datos; if (!D) return;
    const R = D.resultado, esc = escenaBase();
    const fase = $("#b-fase");
    if (i < B.nPasos) {
      const p = D.pasos[i];
      Object.assign(esc, { expandidos: p.ex, frontera: p.fr, actual: p.cur, caminoParcial: p.cam });
      const ultimo = i === B.nPasos - 1;
      if (ultimo && R.exito) esc.solucion = R.camino;
      if (ultimo && !R.exito) esc.mensaje = { titulo: "No encontró camino", sub: "La frontera quedó vacía sin llegar a la fruta.", tono: "mal" };
      fase.className = "insignia busqueda"; fase.textContent = `🔍 Búsqueda · iteración ${p.it}`;
      estadoInterno(p);
      $("#b-explicacion").innerHTML = explicarPaso(p, ultimo);
      graficoLineas($("#b-sparkline"), [{ color: "#f2a600", valores: D.pasos.map((q) => q.tf) }], { cursor: i, alto: 90, area: true });
    } else {
      const j = i - B.nPasos + 1, total = D.cuerpos.length - 1;
      const ultimo = j === total;
      const ultimoPaso = D.pasos[B.nPasos - 1];
      Object.assign(esc, {
        cuerpo: D.cuerpos[j], frutas: ultimo ? [] : esc.frutas,
        expandidos: ultimoPaso ? ultimoPaso.ex : [], tenue: true,
        solucion: R.camino.slice(j), dirCabeza: R.acciones[j - 1],
      });
      if (ultimo) {
        esc.destello = true;
        esc.mensaje = R.seguro
          ? { titulo: "¡Comió la fruta!", sub: "Y todavía tiene camino hasta su cola: estado seguro.", tono: "bien" }
          : { titulo: "Comió, pero quedó en riesgo", sub: "Ya no hay camino desde la cabeza hasta la cola.", tono: "neutro" };
      }
      fase.className = "insignia solucion"; fase.textContent = `🐍 Ejecutando la solución · paso ${j} de ${total}`;
      $("#b-explicacion").innerHTML = ultimo
        ? `<b>Fin.</b> La serpiente siguió las ${total} acciones del plan y su cuerpo creció a L = ${D.cuerpos[j].length}. ` +
          (R.seguro ? "Tras comer, sigue existiendo un camino desde la cabeza hasta la cola, así que no quedó encerrada."
                    : "Tras comer, no existe camino de la cabeza a la cola: este es el caso que el chequeo de seguridad evita.")
        : `<b>Movimiento ${j} de ${total}:</b> la cabeza se mueve «${R.acciones[j - 1]}». El cuerpo avanza y la cola libera su celda (función de transición T).`;
      graficoLineas($("#b-sparkline"), [{ color: "#f2a600", valores: D.pasos.map((q) => q.tf) }], { cursor: B.nPasos - 1, alto: 90, area: true });
    }
    tablero.dibujar(esc);
  }

  function estadoInterno(p) {
    const inf = INFORMADOS.has(B.datos.alg);
    const kpi = (t, v, cls = "") => `<div class="kpi ${cls}"><div class="t">${t}</div><div class="v">${v}</div></div>`;
    $("#b-estado").innerHTML = [
      kpi("Iteración", p.it),
      kpi("Nodo actual", `(${p.cur[0]}, ${p.cur[1]})`, "acento"),
      kpi("Acción que lo generó", p.acc || "inicio"),
      kpi("Profundidad", p.prof),
      kpi("g (costo acumulado)", fmt(p.g, 2)),
      kpi("h (heurística)", inf ? fmt(p.h, 2) : "—"),
      kpi("f (prioridad)", B.datos.alg === "BFS" || B.datos.alg === "DFS" ? (B.datos.alg === "BFS" ? "FIFO" : "LIFO") : fmt(p.f, 2), "acento"),
      kpi("Tamaño frontera", fmt(p.tf)),
      kpi("Generados", fmt(p.gen)),
      kpi("Expandidos", fmt(p.exp)),
    ].join("");
  }

  function explicarPaso(p, ultimo) {
    const alg = B.datos.alg, inf = INFORMADOS.has(alg);
    const nombre = alg === "Beam" ? "Haz" : alg;
    const llegada = p.acc ? `, al que se llegó moviendo «${p.acc}»` : " (estado inicial)";
    const valores = inf
      ? `<span class="eq">g = ${fmt(p.g, 2)} · h = ${fmt(p.h, 2)} · f = ${fmt(p.f, 2)}</span>`
      : `<span class="eq">g = ${fmt(p.g, 2)}</span>`;
    let t = `<b>Iteración ${p.it}.</b> ${nombre}: ${CRITERIO[alg]}. Nodo actual: celda (${p.cur[0]}, ${p.cur[1]})${llegada}. ${valores}. ` +
            `Frontera: ${p.tf} nodos · ${p.exp} expandidos · ${p.gen} generados.`;
    if (ultimo && B.datos.resultado.exito)
      t += ` <b>La cabeza llegó a la fruta:</b> el camino se reconstruye siguiendo a los padres desde este nodo (línea turquesa).`;
    return t;
  }

  VISTAS.busqueda = {
    alMostrar() { tablero.ajustar(); },
    teclado(e) {
      if (e.code === "Space") { e.preventDefault(); repro.alternar(); }
      if (e.code === "ArrowRight") { e.preventDefault(); repro.pausar(); repro.ir(repro.i + 1); }
      if (e.code === "ArrowLeft") { e.preventDefault(); repro.pausar(); repro.ir(repro.i - 1); }
    },
    instancia: () => B.inst,
    resolverCon(alg, heur) {
      B.alg = alg; activarChip($("#b-algoritmos"), "alg", alg);
      if (heur && heur !== "ninguna") $("#b-heur").value = heur;
      actualizarControlesAlg();
      irAVista("busqueda");
      conCarga($("#b-resolver"), resolver);
    },
  };
  generar();
})();

// =====================================================================
// 2. COMPARAR ALGORITMOS
// =====================================================================
(function vistaComparar() {
  const tablero = new Tablero($("#c-canvas"), { maxLado: 260, margenAlto: 0 });
  let filas = null, metrica = "expandidos", ultimaClave = null;
  rango("c-lam", (v) => Number(v).toFixed(1)); rango("c-k");
  chips($("#c-metricas"), "met", (m) => { metrica = m; graficar(); });
  $("#c-correr").addEventListener("click", () => conCarga($("#c-correr"), correr));

  function inst() { return VISTAS.busqueda.instancia(); }
  function vistaPrevia() {
    const e = inst(); if (!e) return;
    tablero.dibujar({ N: e.N, cuerpo: e.cuerpo, frutas: e.fruta ? [e.fruta] : [], muros: e.muros, toroidal: e.toroidal });
    $("#c-info").textContent = `${e.N}×${e.N} · L = ${e.cuerpo.length} · ${e.muros.length} muros. Para cambiarla, edítala en la pestaña 1.`;
  }
  const nombreFila = (f) => (f.algoritmo === "Beam" ? `Haz (k=${$("#c-k").value})` : NOMBRE_ALG[f.algoritmo]);

  async function correr() {
    const e = inst();
    if (!e || !e.fruta) { aviso("La instancia de la pestaña 1 no tiene fruta."); return; }
    const r = await api("comparar", { estado: e, lambda: +$("#c-lam").value, k: +$("#c-k").value });
    filas = r.filas;
    ultimaClave = JSON.stringify(e);
    vistaPrevia(); tabla(); graficar(); mosaico();
  }

  function tabla() {
    const cols = ["longitud", "costo", "generados", "expandidos", "max_frontera", "tiempo_ms"];
    const ok = filas.filter((f) => f.exito);
    const mejor = Object.fromEntries(cols.map((c) => [c, Math.min(...ok.map((f) => f[c]))]));
    const celda = (f, c, d = 0) => `<td class="num ${f.exito && f[c] === mejor[c] ? "mejor" : ""}">${f.exito || !["longitud", "costo"].includes(c) ? fmt(f[c], d) : "—"}</td>`;
    $("#c-tabla tbody").innerHTML = filas.map((f, i) => `
      <tr class="clic" data-i="${i}" title="Ver paso a paso en la pestaña 1">
        <td><b>${nombreFila(f)}</b></td>
        <td>${NOMBRE_HEUR[f.heuristica] || f.heuristica}</td>
        <td class="${f.exito ? "si" : "no"}">${f.exito ? "✔ Sí" : "✘ No"}</td>
        <td class="${f.optimo ? "si" : "no"}">${f.optimo ? "✔ Sí" : "✘ No"}</td>
        ${celda(f, "longitud")}${celda(f, "costo", 2)}${celda(f, "generados")}${celda(f, "expandidos")}${celda(f, "max_frontera")}${celda(f, "tiempo_ms", 3)}
      </tr>`).join("");
    $$("#c-tabla tbody tr").forEach((tr) => tr.addEventListener("click", () => {
      const f = filas[+tr.dataset.i];
      VISTAS.busqueda.resolverCon(f.algoritmo, f.heuristica);
    }));
  }

  function graficar() {
    if (!filas) return;
    const formato = metrica === "tiempo_ms" ? (v) => fmt(v, 3) + " ms" : metrica === "costo" ? (v) => fmt(v, 2) : (v) => fmt(v);
    graficoBarrasH($("#c-grafico"), filas.map((f) => ({
      etiqueta: `${nombreFila(f)}${f.heuristica !== "ninguna" ? " · " + (NOMBRE_HEUR[f.heuristica] || "").split(" ")[0] : ""}`,
      valor: (metrica === "costo" && !f.exito) ? null : f[metrica],
      color: f.optimo ? "#257de0" : "#64748b",
    })), { formato });
  }

  function mosaico() {
    const e = inst();
    const cont = $("#c-mosaico");
    cont.innerHTML = "";
    filas.forEach((f) => {
      const pieza = document.createElement("div");
      pieza.className = "pieza";
      pieza.title = "Ver paso a paso";
      pieza.innerHTML = `<canvas></canvas><div class="nom">${nombreFila(f)} ${f.heuristica !== "ninguna" ? "· " + NOMBRE_HEUR[f.heuristica] : ""}</div>
        <div class="det">${f.exito ? `${f.expandidos} expandidos · costo ${fmt(f.costo, 2)}` : "no llegó"}</div>`;
      cont.append(pieza);
      const t = new Tablero($("canvas", pieza), { fijo: 260 });
      t.dibujar({ N: e.N, cuerpo: e.cuerpo, frutas: e.fruta ? [e.fruta] : [], muros: e.muros, toroidal: e.toroidal,
                  expandidos: f.explorados, frontera: f.frontera, solucion: f.exito ? f.camino : null });
      pieza.addEventListener("click", () => VISTAS.busqueda.resolverCon(f.algoritmo, f.heuristica));
    });
  }

  VISTAS.comparar = {
    alMostrar() {
      tablero.ajustar(); vistaPrevia(); graficar();
      if (filas && ultimaClave !== JSON.stringify(inst()))
        $("#c-info").innerHTML += ' <b style="color:#fde68a">La instancia cambió: vuelve a comparar.</b>';
    },
  };
})();

// =====================================================================
// 3. PARTIDA COMPLETA
// =====================================================================
(function vistaPartida() {
  const P = { alg: "A*", corridas: [], total: 0 };
  const tableros = [new Tablero($("#p-canvas-0"), { maxLado: 640, margenAlto: 330 }), new Tablero($("#p-canvas-1"), { maxLado: 640, margenAlto: 330 })];
  rango("p-N"); rango("p-M"); rango("p-F"); rango("p-lam", (v) => Number(v).toFixed(1));
  chips($("#p-algoritmos"), "alg", (a) => { P.alg = a; $("#p-heur-ctl").style.display = INFORMADOS.has(a) ? "" : "none"; });
  $("#p-dado").addEventListener("click", () => { $("#p-semilla").value = Math.floor(Math.random() * 10000); });
  $("#p-simular").addEventListener("click", () => conCarga($("#p-simular"), simular));
  $("#p-lado").addEventListener("change", () => { $("#p-seg").disabled = $("#p-lado").checked; });

  const repro = new Reproductor($("#p-repro"), {
    fps: 12, alCambiar: (i) => mostrar(i), texto: (i, n) => `Turno ${i} / ${n - 1}`,
  });

  const COLOR_SEG = { true: "#f47230", false: "#257de0" };

  async function simular() {
    const base = {
      N: +$("#p-N").value, semilla: +$("#p-semilla").value, algoritmo: P.alg,
      heuristica: INFORMADOS.has(P.alg) ? $("#p-heur").value : "ninguna",
      lambda: +$("#p-lam").value, muros: +$("#p-M").value, frutas: +$("#p-F").value,
      toroidal: $("#p-toro").checked,
    };
    const lado = $("#p-lado").checked;
    const configs = lado ? [true, false] : [$("#p-seg").checked];
    const res = await Promise.all(configs.map((seg) => api("partida", { ...base, seguridad: seg })));
    P.corridas = res.map((r, k) => preparar(r, configs[k]));
    $("#p-celda-1").classList.toggle("oculto", !lado);
    $("#p-tableros").classList.toggle("dos", lado);
    P.corridas.forEach((c, k) => {
      $("#p-titulo-" + k).textContent = `${NOMBRE_ALG[P.alg]} · ${c.seg ? "con chequeo" : "sin chequeo"}`;
    });
    P.total = Math.max(...P.corridas.map((c) => c.cuerpos.length));
    $("#p-leyenda-graf").innerHTML = P.corridas.map((c) =>
      `<span><i class="m" style="background:${COLOR_SEG[c.seg]}"></i>${c.seg ? "Con chequeo" : "Sin chequeo"}</span>`).join("");
    requestAnimationFrame(() => { tableros.forEach((t) => t.ajustar()); repro.cargar(P.total); });
  }

  function preparar(r, seg) {
    // reconstruye el cuerpo en cada turno con la regla de transición T
    const cuerpos = [r.inicial.cuerpo];
    for (const t of r.turnos) {
      const prev = cuerpos[cuerpos.length - 1];
      cuerpos.push([t.h, ...(t.come ? prev : prev.slice(0, -1))]);
    }
    return { ...r, seg, cuerpos, longitudes: cuerpos.map((c) => c.length) };
  }

  const TEXTO_MODO = { 0: "🍎 Yendo a la fruta", 1: "🌀 Evasión: ruta insegura", 2: "🌀 Evasión: sin camino" };

  function mostrar(i) {
    P.corridas.forEach((c, k) => {
      const ii = Math.min(i, c.cuerpos.length - 1);
      const turno = c.turnos[ii];
      const anterior = c.turnos[ii - 1];
      const fin = ii === c.cuerpos.length - 1;
      const modo = turno ? turno.m : anterior ? anterior.m : 0;
      const esc = {
        N: c.inicial.N, cuerpo: c.cuerpos[ii], muros: c.inicial.muros, toroidal: c.inicial.toroidal,
        frutas: ii === 0 ? c.inicial.frutas : anterior.fr,
        plan: turno ? turno.p : null, planColor: modo > 0 ? COLORES.evasion : COLORES.plan,
        destello: anterior && anterior.come, dirCabeza: anterior ? anterior.a : null,
      };
      if (fin) esc.mensaje = mensajeFin(c.resumen);
      tableros[k].dibujar(esc);
      const insignia = $("#p-modo-" + k);
      insignia.textContent = fin ? "" : TEXTO_MODO[modo];
      insignia.className = "insignia modo " + (modo > 0 ? "evasion" : "ruta");
    });
    marcador(i);
    graficoLineas($("#p-grafico"), P.corridas.map((c) => ({ color: COLOR_SEG[c.seg], valores: c.longitudes })),
      { ymax: P.corridas[0].resumen.meta, cursor: i, alto: 160, refY: P.corridas[0].resumen.meta, etiquetaRef: "tablero lleno" });
    explicar(i);
  }

  function causaCorta(r) {
    if (r.victoria) return "Victoria: tablero lleno";
    if (r.causa_fin.startsWith("Encerrado")) return "Encerrada";
    if (r.causa_fin.startsWith("Límite")) return "200 pasos sin comer";
    return r.causa_fin;
  }

  function mensajeFin(r) {
    if (r.victoria) return { titulo: "¡Victoria! Tablero lleno", sub: `L = ${r.longitud} en ${r.pasos} pasos`, tono: "bien" };
    if (r.causa_fin.startsWith("Encerrado")) return { titulo: "Encerrada", sub: `Sin acciones aplicables con L = ${r.longitud}`, tono: "mal" };
    if (r.causa_fin.startsWith("Límite")) return { titulo: "200 pasos sin comer", sub: `Se quedó dando vueltas con L = ${r.longitud}`, tono: "neutro" };
    return { titulo: r.causa_fin, sub: `L = ${r.longitud}`, tono: "neutro" };
  }

  function marcador(i) {
    const kpi = (t, v, cls = "") => `<div class="kpi ${cls}"><div class="t">${t}</div><div class="v">${v}</div></div>`;
    $("#p-marcador").innerHTML = '<div class="marcador">' + P.corridas.map((c) => {
      const ii = Math.min(i, c.cuerpos.length - 1);
      const r = c.resumen, fin = ii === c.cuerpos.length - 1;
      const G = ii === 0 ? 0 : c.turnos[ii - 1].G;
      const L0 = c.inicial.cuerpo.length;
      return `<div class="marcador-col">${P.corridas.length > 1 ? `<h4 style="color:${COLOR_SEG[c.seg]}">${c.seg ? "Con chequeo" : "Sin chequeo"}</h4>` : ""}
        <div class="kpis">
          ${kpi("Longitud", `${c.cuerpos[ii].length} <small>/ ${r.meta}</small>`, "acento")}
          ${kpi("Frutas", c.cuerpos[ii].length - L0)}
          ${kpi("Pasos", ii)}
          ${kpi("Recompensa G", fmt(G, 1))}
          ${fin ? kpi("Fin", causaCorta(r), "ancho " + (r.victoria ? "bien" : "mal")) : ""}
          ${fin ? kpi("Evasiones de cola", fmt(r.evasiones)) : ""}
          ${fin ? kpi("Nodos exp. / decisión", fmt(r.exp_prom, 1)) : ""}
        </div></div>`;
    }).join("") + "</div>";
  }

  function explicar(i) {
    const c = P.corridas[0];
    const ii = Math.min(i, c.cuerpos.length - 1);
    const t = c.turnos[ii];
    let txt;
    if (ii === c.cuerpos.length - 1) txt = `<b>Fin de la partida:</b> ${c.resumen.causa_fin}. Longitud final ${c.resumen.longitud}, ${c.resumen.frutas} frutas en ${c.resumen.pasos} pasos.`;
    else if (t.m === 0) txt = `<b>Turno ${ii}.</b> La serpiente sigue la ruta que ${NOMBRE_ALG[P.alg]} planeó hasta la fruta (línea verde, ${t.p.length} pasos restantes).` +
      (t.d ? " En este turno <b>planeó una ruta nueva</b>." : "");
    else if (t.m === 1) txt = `<b>Turno ${ii}.</b> Encontró ruta a la fruta, pero al simular el estado tras comer <b>no habría camino hasta su cola</b>. Por eso da un paso siguiendo su cola (línea rosada) y vuelve a planear.`;
    else txt = `<b>Turno ${ii}.</b> No hay ningún camino hasta la fruta. Da un paso siguiendo su cola (línea rosada) para liberar espacio y vuelve a intentar.`;
    if (P.corridas.length > 1) txt += ` <span class="texto-suave">El texto describe el tablero de la izquierda (con chequeo).</span>`;
    $("#p-explicacion").innerHTML = txt;
  }

  VISTAS.partida = {
    alMostrar() { tableros.forEach((t) => t.ajustar()); if (P.corridas.length) mostrar(repro.i); },
    teclado(e) {
      if (e.code === "Space") { e.preventDefault(); repro.alternar(); }
      if (e.code === "ArrowRight") { e.preventDefault(); repro.pausar(); repro.ir(repro.i + 1); }
      if (e.code === "ArrowLeft") { e.preventDefault(); repro.pausar(); repro.ir(repro.i - 1); }
    },
  };
})();

// =====================================================================
// 4. JUEGA CONTRA LA IA
// =====================================================================
(function vistaJugar() {
  const J = { id: null, datos: null, modo: "turnos", dir: null, timer: null, ocupado: false, pista: null, ultimaDir: null };
  const tH = new Tablero($("#j-canvas-h"), { maxLado: 560, margenAlto: 300 });
  const tIA = new Tablero($("#j-canvas-ia"), { maxLado: 560, margenAlto: 300 });
  rango("j-M"); rango("j-vel");
  chips($("#j-modos"), "modo", (m) => { J.modo = m; $("#j-vel-ctl").style.display = m === "real" ? "" : "none"; reiniciarTimer(); });
  $("#j-vel-ctl").style.display = "none";
  $("#j-vel").addEventListener("change", reiniciarTimer);
  $("#j-dado").addEventListener("click", () => { $("#j-semilla").value = Math.floor(Math.random() * 10000); });
  $("#j-nuevo").addEventListener("click", () => conCarga($("#j-nuevo"), nuevo));
  $("#j-pista").addEventListener("click", () => conCarga($("#j-pista"), pista));
  $$(".cruceta button").forEach((b) => b.addEventListener("click", () => pedirDireccion(b.dataset.dir)));

  async function nuevo() {
    const r = await api("jugar/nuevo", { semilla: +$("#j-semilla").value, muros: +$("#j-M").value });
    Object.assign(J, { id: r.id, datos: r, dir: null, pista: null, ultimaDir: null });
    $("#j-mensaje").innerHTML = J.modo === "real"
      ? "Presiona una flecha para arrancar. La serpiente sigue moviéndose sola en la última dirección."
      : "Tu turno: cada tecla mueve tu serpiente un paso y la IA responde con uno suyo.";
    reiniciarTimer();
    dibujar();
  }

  function reiniciarTimer() {
    clearInterval(J.timer); J.timer = null;
    if (J.modo === "real" && J.id) {
      J.timer = setInterval(() => { if (J.dir) mover(J.dir); }, 1000 / +$("#j-vel").value);
    }
  }

  function esReversa(dir) {
    const c = J.datos?.humano.estado.cuerpo;
    if (!c || c.length < 2) return false;
    const N = J.datos.humano.estado.N;
    const d = DIRS[dir];
    let nx = c[0][0] + d[0], ny = c[0][1] + d[1];
    return nx === c[1][0] && ny === c[1][1];
  }

  function pedirDireccion(dir) {
    if (!J.id || J.datos.humano.fin) return;
    if (esReversa(dir)) return;
    $$(".cruceta button").forEach((b) => b.classList.toggle("presionado", b.dataset.dir === dir));
    setTimeout(() => $$(".cruceta button").forEach((b) => b.classList.remove("presionado")), 120);
    J.dir = dir;
    if (J.modo === "turnos") mover(dir);
  }

  async function mover(dir) {
    if (J.ocupado || !J.id || J.datos.humano.fin) return;
    if (esReversa(dir)) dir = J.ultimaDir || dir;
    J.ocupado = true;
    try {
      const r = await api("jugar/paso", { id: J.id, dir });
      J.datos = r; J.pista = null; J.ultimaDir = dir;
      if (r.humano.fin) { clearInterval(J.timer); J.timer = null; finDeJuego(); }
      else $("#j-mensaje").innerHTML = J.modo === "real" ? "¡Sigue así! Come frutas sin chocar." : "Tu turno.";
      dibujar();
    } catch (e) { aviso(e.message); }
    finally { J.ocupado = false; }
  }

  async function pista() {
    if (!J.id) { aviso("Primero inicia una partida."); return; }
    const r = await api("jugar/pista", { id: J.id });
    if (!r.camino.length) { $("#j-mensaje").innerHTML = "A* no encuentra camino a la fruta desde aquí: sigue tu cola para ganar espacio."; return; }
    J.pista = r.camino;
    const dir = direccionEntre(r.camino[0], r.camino[1], J.datos.humano.estado.N);
    $("#j-mensaje").innerHTML = `💡 A* sugiere ir hacia el <b>${dir}</b> (${r.camino.length - 1} pasos hasta la fruta). ` +
      (r.seguro ? "Tras comer aún tendrías salida hacia tu cola." : "<b>Ojo:</b> tras comer te quedarías sin salida hacia tu cola.");
    dibujar();
  }

  function finDeJuego() {
    const h = J.datos.humano, ia = J.datos.ia;
    const dif = h.longitud - ia.longitud;
    $("#j-mensaje").innerHTML = `<b>Fin:</b> ${h.causa}. Terminaste con L = ${h.longitud} en ${h.pasos} pasos; la IA iba en L = ${ia.longitud}. ` +
      (dif > 0 ? "<b>¡Le ganaste a la IA!</b>" : dif === 0 ? "Empate." : `La IA te ganó por ${-dif}.`) + " Presiona <b>Nueva partida</b> para otro intento.";
  }

  function escena(f, extra = {}) {
    const e = f.estado;
    const esc = { N: e.N, cuerpo: e.cuerpo, frutas: e.frutas, muros: e.muros, toroidal: e.toroidal, ...extra };
    if (f.fin) esc.mensaje = f.victoria
      ? { titulo: "¡Tablero lleno!", sub: `L = ${f.longitud}`, tono: "bien" }
      : { titulo: f.causa.startsWith("Colisión") ? "¡Chocaste!" : f.causa.split(" (")[0], sub: `L = ${f.longitud} · ${f.pasos} pasos`, tono: "mal" };
    return esc;
  }
  function dibujar() {
    const d = J.datos; if (!d) return;
    tH.dibujar(escena(d.humano, { sugerencia: J.pista, dirCabeza: J.ultimaDir }));
    tIA.dibujar(escena(d.ia));
    const info = (f) => `L = ${f.longitud} · ${f.frutas} frutas · ${f.pasos} pasos · G = ${fmt(f.G, 0)}`;
    $("#j-h-info").textContent = info(d.humano);
    $("#j-ia-info").textContent = info(d.ia) + (d.ia_modo > 0 ? " · 🌀 siguiendo su cola" : "");
  }

  const TECLAS = { ArrowUp: "norte", KeyW: "norte", ArrowDown: "sur", KeyS: "sur", ArrowLeft: "oeste", KeyA: "oeste", ArrowRight: "este", KeyD: "este" };
  VISTAS.jugar = {
    alMostrar() { tH.ajustar(); tIA.ajustar(); dibujar(); },
    teclado(e) {
      const dir = TECLAS[e.code];
      if (dir) { e.preventDefault(); pedirDireccion(dir); }
    },
  };
  // tablero de muestra antes de jugar
  api("jugar/nuevo", { semilla: 0, muros: 3 }).then((r) => { if (!J.datos) { J.datos = r; J.id = r.id; dibujar(); } }).catch(() => {});
})();

// =====================================================================
// 5. RESULTADOS DEL BENCHMARK
// =====================================================================
(function vistaResultados() {
  let filas = null, metrica = "longitud";
  chips($("#r-metricas"), "met", (m) => { metrica = m; graficar(); });

  async function cargar() {
    const r = await api("benchmark", {});
    if (!r.disponible) {
      $("#r-tabla tbody").innerHTML = '<tr><td class="vacio">No se encontró resultados_benchmark_100_semillas.csv junto a interfaz.py. Corre el benchmark del notebook primero.</td></tr>';
      return;
    }
    filas = r.filas;
    tabla(); graficar();
  }

  function tabla() {
    const enc = `<thead><tr><th>Algoritmo</th><th>Heurística</th><th class="num sin-mayus">λ</th><th class="num">Long. final</th>
      <th class="num">Victorias</th><th class="num">Fin kmáx</th><th class="num">Corte</th><th class="num">Pasos/fruta</th>
      <th class="num">Gen.</th><th class="num">Exp.</th><th class="num">Frontera máx.</th><th class="num">t (ms)</th></tr></thead>`;
    const grupo = (seg) => `<tr class="grupo"><td colspan="12">${seg ? "Con chequeo de seguridad" : "Sin chequeo de seguridad"}</td></tr>` +
      filas.filter((f) => f.chequeo_seguridad === seg).map((f) => `<tr>
        <td><b>${NOMBRE_ALG[f.algoritmo] || f.algoritmo}</b></td><td>${NOMBRE_HEUR[f.heuristica] || f.heuristica}</td>
        <td class="num">${fmt(f.lambda_riesgo, 1)}</td><td class="num">${fmt(f.longitud, 1)} ± ${fmt(f.longitud_std, 1)}</td>
        <td class="num">${fmt(100 * f.victorias)} %</td><td class="num">${fmt(100 * f.fin_kmax)} %</td><td class="num">${fmt(100 * f.corte)} %</td>
        <td class="num">${fmt(f.pasos_fruta, 1)}</td><td class="num">${fmt(f.generados, 1)}</td><td class="num">${fmt(f.expandidos, 1)}</td>
        <td class="num">${fmt(f.frontera, 1)}</td><td class="num">${fmt(f.tiempo_ms, 2)}</td></tr>`).join("");
    $("#r-tabla").innerHTML = enc + "<tbody>" + grupo(false) + grupo(true) + "</tbody>";
  }

  function graficar() {
    if (!filas) return;
    const orden = ["BFS", "DFS", "UCS", "Avara", "A*", "Beam"];
    const elegir = (alg, seg) => filas.find((f) => f.algoritmo === alg && f.chequeo_seguridad === seg &&
      (f.heuristica === "ninguna" || f.heuristica === "manhattan") && (f.lambda_riesgo === 0 || alg === "UCS"));
    const valor = (f) => (f ? (metrica === "victorias" ? 100 * f.victorias : f[metrica]) : null);
    const formato = metrica === "tiempo_ms" ? (v) => fmt(v, 2) : metrica === "victorias" ? (v) => fmt(v) + "%" : (v) => fmt(v, 1);
    graficoBarrasAgrupadas($("#r-grafico"), orden.map((a) => NOMBRE_ALG[a]), [
      { color: "#257de0", valores: orden.map((a) => valor(elegir(a, false))) },
      { color: "#f47230", valores: orden.map((a) => valor(elegir(a, true))) },
    ], { formato, ymax: metrica === "longitud" || metrica === "victorias" ? 100 : undefined });
  }

  VISTAS.resultados = {
    alMostrar() { if (!filas) cargar().catch((e) => aviso(e.message)); else graficar(); },
  };
})();

window.addEventListener("resize", () => VISTAS[vistaActual]?.alMostrar?.());
