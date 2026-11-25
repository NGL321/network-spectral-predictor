"""Visual layer: nominal-vs-actual players, film strips and inspection reports.

Metrology vocabulary, inverted on purpose: the *actual* world is the deterministically generated one; the
*nominal* world is the model's prediction. Deviation = actual - nominal, drawn as a whisker between the two
bobs, magnified like a deviation plot and coloured against a tolerance: green in, orange near (within twice
the tolerance), vermillion out.

Needs matplotlib and IPython (the `notebooks` extra). Players are self-contained HTML + JavaScript: they run in
Jupyter (once the notebook has been executed locally), VS Code and the HTML exports. Film strips are static
images that render everywhere, GitHub included.
"""

from __future__ import annotations

import json
import uuid

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from IPython.display import HTML, Markdown
from matplotlib.patches import Circle

COLORS = {
    "actual": "#1F2328",
    "nominal": "#0072B2",
    "in": "#009E73",
    "near": "#E69F00",
    "out": "#D55E00",
    "train": "#CC79A7",
    "test": "#56B4E9",
    "spring": "#6E7781",
    "off": "#C9D1D9",
    "muted": "#8C959F",
    "band": "#009E73",
}
SERIES = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
STATUS_TEXT = {"in": "✓ **in**", "near": "△ **near**", "out": "✗ **out**", None: "—"}


def style():
    """House style for every figure: light grid, left-aligned titles, colour-blind-safe series (Okabe-Ito)."""
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "figure.figsize": (10, 4),
        "figure.constrained_layout.use": True, "figure.facecolor": "white",
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 11, "axes.labelcolor": "#1F2328", "axes.edgecolor": "#8C959F",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#E6E8EB", "grid.linewidth": 0.8,
        "axes.prop_cycle": cycler(color=SERIES), "legend.frameon": False, "legend.fontsize": 10,
        "xtick.color": "#57606A", "ytick.color": "#57606A", "lines.linewidth": 1.8,
    })


_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def fmt(x, sig: int = 3) -> str:
    """Compact number: scientific (with a superscript exponent) when tiny or huge."""
    if x is None:
        return "—"
    if isinstance(x, str):
        return x
    x = float(x)
    if not np.isfinite(x):
        return "∞" if x > 0 else ("−∞" if x < 0 else "—")
    if x == 0:
        return "0"
    a = abs(x)
    if a < 1e-3 or a >= 1e6:
        m, e = f"{x:.{sig - 1}e}".split("e")
        return f"{m} × 10{str(int(e)).translate(_SUP)}".replace("-", "−")
    if a >= 10 ** (sig - 1):
        return f"{x:,.0f}".replace("-", "−")
    return f"{x:.{sig}g}".replace("-", "−")


def pct(x: float, sig: int = 2) -> str:
    return fmt(100 * x, sig) + "%"


def status(deviation: float, tolerance: float) -> str:
    a = abs(deviation)
    return "in" if a <= tolerance else ("near" if a <= 2 * tolerance else "out")


# ---------------------------------------------------------------------------------------------- the row drawing

def _zigzag(p0, p1, teeth: int = 6, width: float = 0.045):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    v = p1 - p0
    length = np.linalg.norm(v)
    if length < 1e-9:
        return np.array([p0, p1])
    t = v / length
    nrm = np.array([-t[1], t[0]])
    pts = [p0, p0 + 0.18 * v]
    s = np.linspace(0.18, 0.82, 2 * teeth + 1)[1:-1]
    for j, sj in enumerate(s):
        pts.append(p0 + sj * v + (1 if j % 2 == 0 else -1) * width * nrm)
    pts += [p0 + 0.82 * v, p1]
    return np.array(pts)


def draw_row(ax, theta_act, theta_nom=None, edges=(), gates=None, *, amplify: float = 1.0, tol: float | None = None,
             rod: float = 1.0, spacing: float = 1.0, attach: float = 0.45, bob_size: float = 0.11,
             nominal_gates=None, limits=None):
    """Draw planar pendulums in a row: actual bobs filled, nominal bobs hollow with the deviation magnified."""
    theta_act = np.asarray(theta_act, float)
    n = len(theta_act)
    px = np.arange(n) * spacing

    def bob(i, th, frac=1.0):
        return np.array([px[i] + rod * frac * np.sin(th), -rod * frac * np.cos(th)])

    ax.plot([px[0] - 0.45 * spacing, px[-1] + 0.45 * spacing], [0, 0], color=COLORS["actual"], lw=2.4,
            solid_capstyle="round", zorder=2)
    for k, (u, v) in enumerate(edges):
        on = gates is None or gates[k] > 0.5
        kw = dict(color=COLORS["spring"] if on else COLORS["off"], lw=1.4 if on else 1.1,
                  ls="-" if on else (0, (3, 2.5)), zorder=1)
        if abs(u - v) == 1:
            z = _zigzag(bob(u, theta_act[u], attach), bob(v, theta_act[v], attach))
            ax.plot(z[:, 0], z[:, 1], **kw)
        else:
            x0, x1 = px[u], px[v]
            h = 0.06 + 0.05 * abs(u - v)
            xs = np.linspace(0, 1, 40)
            ax.plot(x0 + (x1 - x0) * xs, 0.02 + 4 * h * xs * (1 - xs), **kw)
        if nominal_gates is not None and (nominal_gates[k] > 0.5) != on and abs(u - v) == 1:
            z = _zigzag(bob(u, theta_act[u], attach + 0.1), bob(v, theta_act[v], attach + 0.1))
            ax.plot(z[:, 0], z[:, 1], color=COLORS["nominal"], lw=1.0, ls=(0, (2, 2)), zorder=1)
    if theta_nom is not None:
        dev = np.asarray(theta_nom, float) - theta_act
        for i in range(n):
            b_act, b_nom = bob(i, theta_act[i]), bob(i, theta_act[i] + amplify * dev[i])
            ax.plot([px[i], b_nom[0]], [0, b_nom[1]], color=COLORS["nominal"], lw=1.0, ls=(0, (3, 2)), zorder=3)
            c = COLORS[status(dev[i], tol)] if tol is not None else COLORS["muted"]
            ax.plot([b_act[0], b_nom[0]], [b_act[1], b_nom[1]], color=c, lw=2.2, zorder=5, solid_capstyle="round")
            ax.add_patch(Circle(b_nom, bob_size, fill=False, ec=COLORS["nominal"], lw=1.8, zorder=6))
    for i in range(n):
        b = bob(i, theta_act[i])
        ax.plot([px[i], b[0]], [0, b[1]], color=COLORS["actual"], lw=2.0, zorder=4)
        ax.add_patch(Circle(b, bob_size, color=COLORS["actual"], zorder=5))
        ax.plot(px[i], 0, "o", ms=3.5, color="white", mec=COLORS["actual"], mew=1.2, zorder=6)
    ax.set_aspect("equal")
    ax.axis("off")
    if limits is None:
        limits = row_limits(n, edges, {"rod": rod, "bob": bob_size, "rise": rod * max(0.0, -np.cos(
            min(np.abs(theta_act).max(), np.pi)))}, spacing)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])


def geometry(theta_max: float, spacing: float = 1.0) -> dict:
    """Rod length, bob radius and rise above the bar that keep each pendulum in its own lane.

    A bob reaches rod * sin(min(theta_max, pi/2)) sideways. Holding that to 0.38 of the spacing stops two
    neighbours swinging towards each other from overlapping. Past pi/2 the bob rises above the bar by
    rod * (-cos theta_max), so the drawing needs headroom.
    """
    reach = np.sin(min(float(theta_max), np.pi / 2))
    rod = float(min(1.0, 0.38 * spacing / max(reach, 1e-9)))
    return {"rod": rod, "bob": 0.11 if rod > 0.7 else 0.09,
            "rise": rod * max(0.0, -np.cos(min(float(theta_max), np.pi)))}


def _arc_top(edges) -> float:
    """Height of the tallest arc drawn for a spring between non-neighbours (0.1 when there are none)."""
    far = [abs(u - v) for u, v in edges if abs(u - v) > 1]
    return 0.02 + 0.06 + 0.05 * max(far) + 0.04 if far else 0.1


def row_limits(n: int, edges, geo: dict, spacing: float = 1.0):
    """Axis limits that fit n pendulums of the given geometry, their arcs and their swings."""
    top = max(_arc_top(edges), geo["rise"] + geo["bob"] + 0.06, 0.25)
    return ((-0.7 * spacing, (n - 1) * spacing + 0.7 * spacing), (-(geo["rod"] + geo["bob"] + 0.07), top))


def auto_amplify(max_dev: float, target: float = 0.3) -> float:
    """Largest power of ten (>= 1) that keeps the magnified deviation under `target` rad."""
    if max_dev <= 0:
        return 1.0
    amp = 1.0
    while amp * 10 * max_dev <= target and amp < 1e6:
        amp *= 10
    return amp


def film_strip(theta_act, theta_nom, edges, gates=None, *, dt: float, frames: int = 6, ncols: int = 3,
               amplify: float | None = None, tol: float | None = None, title: str | None = None,
               switch_steps=(), times=None):
    """Static frames of actual vs nominal, plus the deviation trace with the frames marked."""
    theta_act, theta_nom = np.asarray(theta_act), np.asarray(theta_nom)
    T, n = theta_act.shape
    dev = np.abs(theta_nom - theta_act).max(1)
    amplify = auto_amplify(dev.max()) if amplify is None else amplify
    idx = (np.linspace(0, T - 1, frames).round().astype(int) if times is None
           else np.asarray([min(T - 1, int(round(t / dt))) for t in times]))
    rows = int(np.ceil(len(idx) / ncols))
    geo = geometry(np.abs(theta_act).max() + min(0.3, amplify * dev.max()))
    xlim, ylim = row_limits(n, edges, geo)
    panel_h = (11 / ncols) * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]) + 0.35
    fig = plt.figure(figsize=(11, rows * panel_h + 2.1))
    gs = fig.add_gridspec(rows + 1, ncols, height_ratios=[panel_h] * rows + [1.5])
    for j, t in enumerate(idx):
        ax = fig.add_subplot(gs[j // ncols, j % ncols])
        g = None if gates is None else (gates[t] if np.ndim(gates) == 2 else gates)
        draw_row(ax, theta_act[t], theta_nom[t], edges, g, amplify=amplify, tol=tol, rod=geo["rod"],
                 bob_size=geo["bob"], limits=(xlim, ylim))
        ax.set_title(f"t = {round(t * dt, 3):g} s", fontsize=10, loc="left", color="#57606A", fontweight="normal")
    axd = fig.add_subplot(gs[rows, :])
    tt = np.arange(T) * dt
    if tol is not None:
        axd.axhspan(0, tol, color=COLORS["band"], alpha=0.12, lw=0)
        axd.axhline(tol, color=COLORS["band"], lw=1, ls="--")
        axd.text(tt[-1], tol, "  tolerance", va="center", ha="left", fontsize=9, color=COLORS["in"])
    axd.plot(tt, dev, color=COLORS["actual"], lw=1.3)
    for t in idx:
        axd.axvline(t * dt, color=COLORS["nominal"], lw=0.8, alpha=0.5)
    for s in switch_steps:
        axd.axvline(s * dt, color=COLORS["out"], lw=1.2, ls=":")
    axd.set_xlim(0, tt[-1])
    axd.set_ylim(0, max(dev.max() * 1.1, (tol or 0) * 1.6, 1e-12))
    axd.set_xlabel("time (s)")
    axd.set_ylabel("max |deviation| (rad)")
    sub = f"actual filled, nominal hollow; deviation magnified ×{amplify:g}"
    fig.suptitle((title + "\n" if title else "") + sub, x=0.01, ha="left", fontsize=12)
    return fig


# ------------------------------------------------------------------------------------------------- the player

_PLAYER_HTML = r"""
<div id="__UID__" style="font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;max-width:940px;
 border:1px solid #d0d7de;border-radius:10px;padding:12px 14px 10px;background:#ffffff;color:#1f2328;margin:8px 0 16px;">
 <div style="display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap;">
  <div style="font-weight:600;font-size:15px;">__TITLE__</div>
  <div id="__UID__-badge" style="font-size:12px;font-weight:600;padding:2px 10px;border-radius:999px;">IN TOLERANCE</div>
 </div>
 <div style="font-size:12.5px;color:#57606a;margin:3px 0 6px;">__SUBTITLE__</div>
 <canvas id="__UID__-world" style="width:100%;height:__HEIGHT__px;display:block;"></canvas>
 <canvas id="__UID__-trace" style="width:100%;height:104px;display:block;margin-top:2px;"></canvas>
 <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:8px;font-size:13px;">
  <button id="__UID__-play" style="font-size:13px;padding:4px 14px;border:1px solid #d0d7de;border-radius:6px;
   background:#f6f8fa;color:#1f2328;cursor:pointer;min-width:84px;">&#9654; Play</button>
  <input id="__UID__-slider" type="range" min="0" max="__TMAX__" value="0" step="1" style="flex:1;min-width:200px;">
  <span id="__UID__-time" style="font-variant-numeric:tabular-nums;min-width:250px;color:#1f2328;"></span>
 </div>
 <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-top:6px;font-size:12.5px;color:#57606a;">
  <label>deviation &times; <select id="__UID__-amp" style="font-size:12.5px;">__AMPS__</select></label>
  <label>speed <select id="__UID__-speed" style="font-size:12.5px;"><option value="0.25">&frac14;&times;</option>
   <option value="0.5">&frac12;&times;</option><option value="1" selected>1&times; (real time)</option>
   <option value="2">2&times;</option></select></label>
  <span><b style="color:#1f2328;">&#9679;</b> actual (generated) &nbsp; <b style="color:#0072B2;">&#9675;</b> nominal (predicted)
   &nbsp; whisker = deviation: <b style="color:#009E73;">in</b> / <b style="color:#E69F00;">near</b> /
   <b style="color:#D55E00;">out</b> of &plusmn;__TOL__ rad__GATELEGEND__</span>
 </div>
</div>
<script>
(function(){
const D = __DATA__;
const U = "__UID__";
const el = s => document.getElementById(U + s);
const world = el("-world"), trace = el("-trace"), slider = el("-slider"), play = el("-play"),
      tlab = el("-time"), badge = el("-badge"), ampSel = el("-amp"), speedSel = el("-speed");
const C = {act:"#1f2328", nom:"#0072B2", in:"#009E73", near:"#E69F00", out:"#D55E00", spring:"#6e7781",
           off:"#c9d1d9", band:"rgba(0,158,115,0.13)", grid:"#e6e8eb", muted:"#57606a"};
const T = D.act.length, n = D.act[0].length;
let frame = 0, playing = false, last = null, acc = 0, amp = D.amp;
const maxdev = D.dev.map(r => r.reduce((m, v) => Math.max(m, Math.abs(v)), 0));
const devmax = Math.max(1e-15, ...maxdev);
let thmax = 0; D.act.forEach(r => r.forEach(v => { thmax = Math.max(thmax, Math.abs(v)); }));
const lift = Math.max(0, -Math.cos(Math.min(thmax, Math.PI)));
const status = a => a <= D.tol ? "in" : (a <= 2 * D.tol ? "near" : "out");
const fmt = v => (v === 0 ? "0" : (Math.abs(v) < 0.01 ? v.toExponential(1) : v.toPrecision(3)));
const gatesAt = (G, t) => G ? (Array.isArray(G[0]) ? G[t] : G) : null;
function setup(c) {
  const dpr = window.devicePixelRatio || 1, w = c.clientWidth || 880, h = c.clientHeight || 100;
  c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
  const ctx = c.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); return [ctx, w, h];
}
function zigzag(ctx, x0, y0, x1, y1, teeth, w) {
  const dx = x1 - x0, dy = y1 - y0, L = Math.hypot(dx, dy) || 1, nx = -dy / L, ny = dx / L;
  ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0 + 0.18 * dx, y0 + 0.18 * dy);
  const m = 2 * teeth;
  for (let j = 1; j < m; j++) { const s = 0.18 + 0.64 * j / m, sg = (j % 2 ? 1 : -1);
    ctx.lineTo(x0 + s * dx + sg * w * nx, y0 + s * dy + sg * w * ny); }
  ctx.lineTo(x0 + 0.82 * dx, y0 + 0.82 * dy); ctx.lineTo(x1, y1); ctx.stroke();
}
function drawWorld() {
  const [ctx, W, H] = setup(world); ctx.clearRect(0, 0, W, H);
  const sp = W / (n + 1), arcRoom = 22, reach = Math.max(Math.sin(Math.min(thmax, Math.PI / 2)), 0.05);
  const rod = Math.min((H - arcRoom - 26) / (1 + lift), sp * 2.2, (0.46 * sp - 10) / reach);
  const off = Math.max(0, (H - (arcRoom + 8 + rod * (1 + lift) + 14)) / 2);
  const top = off + arcRoom + 8 + rod * lift, px = i => sp * (i + 1);
  const a = D.act[frame], dv = D.dev[frame], g = gatesAt(D.gates, frame), gn = gatesAt(D.nom_gates, frame);
  const bob = (i, th, f) => [px(i) + rod * (f || 1) * Math.sin(th), top + rod * (f || 1) * Math.cos(th)];
  ctx.lineCap = "round";
  D.edges.forEach((e, k) => {
    const u = e[0], v = e[1], on = !g || g[k] > 0.5;
    ctx.save(); ctx.strokeStyle = on ? C.spring : C.off; ctx.lineWidth = on ? 1.7 : 1.3;
    ctx.setLineDash(on ? [] : [5, 4]);
    if (Math.abs(u - v) === 1) { const p = bob(u, a[u], D.attach), q = bob(v, a[v], D.attach);
      zigzag(ctx, p[0], p[1], q[0], q[1], 7, 5); }
    else { const x0 = px(u), x1 = px(v), hh = Math.min(arcRoom, 5 + 3 * Math.abs(u - v));
      ctx.beginPath(); ctx.moveTo(x0, top - 3); ctx.quadraticCurveTo((x0 + x1) / 2, top - 3 - 2 * hh, x1, top - 3); ctx.stroke(); }
    if (gn && (gn[k] > 0.5) !== on) {
      ctx.strokeStyle = C.nom; ctx.lineWidth = 1.4; ctx.setLineDash([3, 3]);
      if (Math.abs(u - v) === 1) { const p = bob(u, a[u], D.attach + 0.12), q = bob(v, a[v], D.attach + 0.12);
        zigzag(ctx, p[0], p[1], q[0], q[1], 7, 4); }
      else { const x0 = px(u), x1 = px(v), hh = Math.min(arcRoom, 5 + 3 * Math.abs(u - v)) + 5;
        ctx.beginPath(); ctx.moveTo(x0, top - 3); ctx.quadraticCurveTo((x0 + x1) / 2, top - 3 - 2 * hh, x1, top - 3); ctx.stroke(); } }
    ctx.restore();
  });
  ctx.strokeStyle = C.act; ctx.lineWidth = 3; ctx.beginPath();
  ctx.moveTo(px(0) - sp * 0.45, top); ctx.lineTo(px(n - 1) + sp * 0.45, top); ctx.stroke();
  for (let i = 0; i < n; i++) { const b = bob(i, a[i] + amp * dv[i]);
    ctx.strokeStyle = C.nom; ctx.lineWidth = 1.2; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(px(i), top); ctx.lineTo(b[0], b[1]); ctx.stroke(); ctx.setLineDash([]); }
  for (let i = 0; i < n; i++) { const b = bob(i, a[i]);
    ctx.strokeStyle = C.act; ctx.lineWidth = 2.4; ctx.beginPath(); ctx.moveTo(px(i), top); ctx.lineTo(b[0], b[1]); ctx.stroke();
    ctx.fillStyle = C.act; ctx.beginPath(); ctx.arc(b[0], b[1], 9, 0, 2 * Math.PI); ctx.fill(); }
  for (let i = 0; i < n; i++) { const b = bob(i, a[i] + amp * dv[i]), ba = bob(i, a[i]);
    if (Math.hypot(b[0] - ba[0], b[1] - ba[1]) > 1.5) { ctx.strokeStyle = C[status(Math.abs(dv[i]))]; ctx.lineWidth = 2.6;
      ctx.beginPath(); ctx.moveTo(ba[0], ba[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); }
    ctx.strokeStyle = C.nom; ctx.lineWidth = 2.2; ctx.beginPath(); ctx.arc(b[0], b[1], 9, 0, 2 * Math.PI); ctx.stroke();
    ctx.fillStyle = "#fff"; ctx.strokeStyle = C.act; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.arc(px(i), top, 3.2, 0, 2 * Math.PI); ctx.fill(); ctx.stroke(); }
}
function drawTrace() {
  const [ctx, W, H] = setup(trace); ctx.clearRect(0, 0, W, H);
  const l = 66, r = 12, t = 8, b = 24, w = W - l - r, h = H - t - b;
  const ymax = Math.max(devmax * 1.08, D.tol * 1.6), X = k => l + w * k / Math.max(1, T - 1), Y = v => t + h * (1 - v / ymax);
  ctx.fillStyle = C.band; ctx.fillRect(l, Y(Math.min(D.tol, ymax)), w, Y(0) - Y(Math.min(D.tol, ymax)));
  ctx.strokeStyle = C.grid; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(l, t + h); ctx.lineTo(l + w, t + h); ctx.stroke();
  ctx.fillStyle = C.muted; ctx.font = "11px -apple-system,Segoe UI,Roboto,sans-serif"; ctx.textAlign = "right";
  ctx.fillText(fmt(ymax), l - 6, t + 8); ctx.fillText("0", l - 6, t + h + 3);
  if (D.tol < ymax && Y(0) - Y(D.tol) > 16) { ctx.fillStyle = C.in; ctx.fillText("tolerance", l - 6, Y(D.tol) + 4); }
  ctx.fillStyle = C.muted; ctx.textAlign = "left"; ctx.fillText("max |deviation| (rad)", l + 4, t + 9);
  ctx.textAlign = "center"; const tend = (T - 1) * D.dt, step = tend > 40 ? 10 : (tend > 12 ? 2 : 1);
  for (let s = 0; s <= tend + 1e-9; s += step) { ctx.fillText(s + " s", X(s / D.dt), t + h + 16); }
  (D.switches || []).forEach(k => { ctx.strokeStyle = C.out; ctx.setLineDash([3, 3]); ctx.beginPath();
    ctx.moveTo(X(k), t); ctx.lineTo(X(k), t + h); ctx.stroke(); ctx.setLineDash([]); });
  ctx.strokeStyle = C.act; ctx.lineWidth = 1.4; ctx.beginPath();
  maxdev.forEach((v, k) => { const x = X(k), y = Y(v); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke();
  ctx.strokeStyle = C.nom; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.moveTo(X(frame), t); ctx.lineTo(X(frame), t + h); ctx.stroke();
}
function update() {
  drawWorld(); drawTrace(); slider.value = frame;
  const md = maxdev[frame], s = status(md);
  tlab.textContent = "t = " + (frame * D.dt).toFixed(2) + " s   max |deviation| " + fmt(md) + " rad";
  const look = {in: ["IN TOLERANCE", "#e6f4ea", "#05603a"], near: ["NEAR LIMIT", "#fff4e0", "#8a5300"],
                out: ["OUT OF TOLERANCE", "#fde8e1", "#9a3412"]}[s];
  badge.textContent = look[0]; badge.style.background = look[1]; badge.style.color = look[2];
}
function tick(ts) {
  if (!playing) return; if (last === null) last = ts;
  acc += (ts - last) / 1000 * parseFloat(speedSel.value) / D.dt; last = ts;
  const adv = Math.floor(acc); if (adv > 0) { acc -= adv; frame = (frame + adv) % T; update(); }
  requestAnimationFrame(tick);
}
play.onclick = () => { playing = !playing; play.innerHTML = playing ? "&#10074;&#10074; Pause" : "&#9654; Play";
  last = null; acc = 0; if (playing) requestAnimationFrame(tick); };
slider.oninput = () => { frame = parseInt(slider.value); update(); };
ampSel.onchange = () => { amp = parseFloat(ampSel.value); update(); };
window.addEventListener("resize", update);
update();
})();
</script>
"""


def _round_rows(a: np.ndarray, sig: int) -> list:
    return [[float(f"{v:.{sig}g}") for v in row] for row in np.asarray(a, float)]


def player(theta_act, theta_nom, edges, gates=None, *, dt: float, tol: float, title: str = "",
           subtitle: str = "", amplify: float | None = None, nominal_gates=None, switch_steps=(),
           height: int = 280, attach: float = 0.45, max_frames: int = 1200) -> HTML:
    """Interactive nominal-vs-actual player: play or scrub through time, magnify the deviation.

    theta_act, theta_nom: (T, n) angles in rad. gates: (m,) or (T, m) actual gate values; nominal_gates likewise,
    drawn only where they differ from the actual ones. tol: tolerance (rad) for the deviation colours.
    """
    theta_act, theta_nom = np.asarray(theta_act, float), np.asarray(theta_nom, float)
    stride = max(1, int(np.ceil(len(theta_act) / max_frames)))
    act, nom = theta_act[::stride], theta_nom[::stride]
    dev = nom - act
    amp = auto_amplify(np.abs(dev).max()) if amplify is None else amplify
    amps = sorted({1, 10, 100, 1000, 10000, 100000, amp})

    def sub(G):
        if G is None:
            return None
        G = np.asarray(G, float)
        return G[::stride].tolist() if G.ndim == 2 else G.tolist()

    data = {"dt": dt * stride, "act": _round_rows(act, 7), "dev": _round_rows(dev, 4), "edges": [list(e) for e in edges],
            "gates": sub(gates), "nom_gates": sub(nominal_gates), "tol": tol, "amp": amp, "attach": attach,
            "switches": [s / stride for s in switch_steps]}
    thmax = float(np.abs(act).max())
    reach = max(np.sin(min(thmax, np.pi / 2)), 0.05)
    lift = max(0.0, -np.cos(min(thmax, np.pi)))
    sp = 910 / (act.shape[1] + 1)
    rod = min((height - 48) / (1 + lift), 2.2 * sp, (0.46 * sp - 10) / reach)
    height = int(min(height, max(170, 22 + 8 + rod * (1 + lift) + 40)))
    uid = "nsp" + uuid.uuid4().hex[:10]
    html = (_PLAYER_HTML.replace("__UID__", uid).replace("__TITLE__", title).replace("__SUBTITLE__", subtitle)
            .replace("__GATELEGEND__", " &nbsp; <b style='color:#0072B2;'>- -</b> where the nominal world's springs differ"
                     if nominal_gates is not None else "")
            .replace("__HEIGHT__", str(height)).replace("__TMAX__", str(len(act) - 1)).replace("__TOL__", fmt(tol))
            .replace("__AMPS__", "".join(f'<option value="{a:g}"{" selected" if a == amp else ""}>{a:g}</option>'
                                         for a in amps))
            .replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    return HTML(html)


# ----------------------------------------------------------------------------------------------- report tables

def inspection_report(rows: list[dict], title: str = "Inspection report") -> Markdown:
    """A CMM-style report. Each row: characteristic, nominal, actual, deviation, tolerance, status (or omit)."""
    lines = [f"**{title}**", "",
             "| Characteristic | Nominal (predicted) | Actual (generated) | Deviation | Tolerance | Status |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        st = r.get("status")
        if st is None and isinstance(r.get("deviation"), (int, float)) and isinstance(r.get("tolerance"), (int, float)):
            st = status(r["deviation"], r["tolerance"])
        lines.append("| " + " | ".join([str(r.get("characteristic", "")),
                                         _cell(r.get("nominal")), _cell(r.get("actual")), _cell(r.get("deviation")),
                                         _cell(r.get("tolerance"), pm=True), STATUS_TEXT.get(st, str(st))]) + " |")
    return Markdown("\n".join(lines))


def _cell(v, pm: bool = False) -> str:
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    return ("±" if pm else "") + fmt(v, 4)


def ledger(entries: list[tuple[str, object, str]], title: str = "Configuration ledger") -> Markdown:
    """Table of (setting, value, meaning)."""
    lines = [f"**{title}**", "", "| Setting | Value | Meaning |", "| --- | --- | --- |"]
    for k, v, meaning in entries:
        lines.append(f"| `{k}` | {v} | {meaning} |")
    return Markdown("\n".join(lines))


def finding(text: str) -> Markdown:
    """A one-paragraph callout for the main result of a section."""
    return Markdown(f"> **Finding.** {text}")


def thetas(X: np.ndarray) -> np.ndarray:
    """Planar angles from states (..., 2n) -> (..., n)."""
    return np.asarray(X)[..., 0::2]
