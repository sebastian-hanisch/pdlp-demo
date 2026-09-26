"""Plotly-Abbildungen: Weg von PDHG und PDLP (n = 2), Residuenkurven mit Neustarts, Schrittweite und Primalgewicht, Ablation der fünf Bausteine, Genauigkeitsleiter, Nichtnullen (Ecke), Operationen gegen Simplex
und Innere Punkte, große dünne Instanzen, Skalierung, Klee-Minty-Würfel. Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen; bei gleichem Achsenmaßstab (scaleanchor) gibt es keine expliziten Bereiche."""

import numpy as np
import plotly.graph_objects as go

import pdl_scenario as S

TEAL, ORANGE, RED, BLUE, GREY, PURPLE = "#2F6B65", "#e8a13a", "#d62728", "#1f4e9c", "#8a8f98", "#7b3fbf"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height, legend_y=-0.25):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=legend_y), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _box(inst):
    """Obere Schranken je Dienst aus den <=-Zeilen mit nichtnegativen Koeffizienten; ohne Schranke das 1.5-Fache der größten."""
    A, b, _c = inst.arrays()
    U = np.full(inst.n, np.inf)
    for i, s in enumerate(inst.senses):
        if s != S.LE or np.any(A[i] < 0):
            continue
        for j in range(inst.n):
            if A[i, j] > 0:
                U[j] = min(U[j], b[i] / A[i, j])
    finite = U[np.isfinite(U)]
    fill = 1.5 * (finite.max() if len(finite) else 10.0)
    return np.where(np.isfinite(U), U, fill)


def feasible_polygon(inst):
    """Zulässiges Vieleck einer Instanz mit zwei Variablen (Kasten [0, U], an <=- und >=-Zeilen abgeschnitten; Gleichungen bleiben unberücksichtigt)."""
    U = _box(inst)
    poly = [np.array([0.0, 0.0]), np.array([U[0], 0.0]), np.array([U[0], U[1]]), np.array([0.0, U[1]])]
    A, b, _c = inst.arrays()
    for i, s in enumerate(inst.senses):
        if s == S.EQ:
            continue
        a, beta = (A[i], b[i]) if s == S.LE else (-A[i], -b[i])
        out = []
        for k, p in enumerate(poly):
            q = poly[(k + 1) % len(poly)]
            sp, sq = float(a @ p) - beta, float(a @ q) - beta
            if sp <= 0:
                out.append(p)
            if (sp < 0 < sq) or (sq < 0 < sp):
                out.append(p + (q - p) * (sp / (sp - sq)))
        poly = out
        if not poly:
            break
    return poly


def build_path(a, k):
    """Zwei Variablen: zulässige Menge, bisherige Iterierte, Mittelwert (falls aktiv), Neustart-Marken, aktuelle Iterierte und der Optimalpunkt des Simplex (Koordinaten des Originals)."""
    inst, res = a.inst, a.res
    poly = feasible_polygon(inst)
    fig = go.Figure()
    if poly:
        px, py = [p[0] for p in poly], [p[1] for p in poly]
        fig.add_trace(go.Scatter(x=px + [px[0]], y=py + [py[0]], fill="toself", fillcolor="rgba(47,107,101,0.18)", line=dict(color=TEAL, width=2), name="zulässige Menge", hoverinfo="skip"))
    pts = np.array(res.points[: k + 1])
    fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines+markers", line=dict(color=ORANGE, width=2), marker=dict(size=5, color=ORANGE), name=f"Iterierte 0 bis {k}", hoverinfo="skip"))
    if len(res.avg_points):
        avg = np.array(res.avg_points[: k + 1])
        fig.add_trace(go.Scatter(x=avg[:, 0], y=avg[:, 1], mode="lines", line=dict(color=PURPLE, width=2, dash="dot"), name="Mittelwert", hoverinfo="skip"))
    marks = [i for i in res.restart_iters if i <= k]
    if marks:
        rp = np.array([res.points[i] for i in marks])
        fig.add_trace(go.Scatter(x=rp[:, 0], y=rp[:, 1], mode="markers", marker=dict(size=10, color=BLUE, symbol="diamond"), name="Neustarts", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[pts[-1, 0]], y=[pts[-1, 1]], mode="markers", marker=dict(size=13, color=RED, line=dict(color="white", width=1)), name="aktuelle Iterierte", hoverinfo="skip"))
    if a.simplex.status == "optimal":
        xs = np.array(a.simplex.x)
        fig.add_trace(go.Scatter(x=[xs[0]], y=[xs[1]], mode="markers", marker=dict(size=14, color=BLUE, symbol="star"), name="Optimum (Simplex)", hoverinfo="skip"))
    fig.update_xaxes(title_text="Dienst 1", scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title_text="Dienst 2")
    return _base(fig, 460, legend_y=-0.3)


def build_convergence(res, basic=None):
    """Relative Residuen (primal, dual, Lücke) des Kandidaten über die Iterationen (log-y), Neustarts als Marken; grau gestrichelt: das größte Residuum der Grundform."""
    its = np.array(res.check_iters)
    fig = go.Figure()
    for name, ys, color, dash in (("primales Residuum", res.prim_res, TEAL, "solid"), ("duales Residuum", res.dual_res, PURPLE, "solid"), ("relative Lücke", res.gap, BLUE, "solid")):
        fig.add_trace(go.Scatter(x=its, y=np.maximum(np.array(ys, dtype=float), 1e-20), mode="lines", line=dict(color=color, width=2, dash=dash), name=name))
    if basic is not None and basic is not res and len(basic.check_iters):
        fig.add_trace(go.Scatter(x=basic.check_iters, y=np.maximum(np.array([max(a, b, c) for a, b, c in zip(basic.prim_res, basic.dual_res, basic.gap)]), 1e-20), mode="lines", line=dict(color=GREY, width=2, dash="dash"),
                                 name="Grundform (größtes Residuum)"))
    if res.restart_iters:
        idx = {k: i for i, k in enumerate(res.check_iters)}
        rk = [k for k in res.restart_iters if k in idx]
        fig.add_trace(go.Scatter(x=rk, y=[max(res.prim_res[idx[k]], res.dual_res[idx[k]], res.gap[idx[k]], 1e-20) for k in rk], mode="markers", marker=dict(size=9, color=ORANGE, symbol="diamond"), name=f"{len(rk)} Neustarts"))
    fig.update_yaxes(type="log", title_text="relatives Residuum")
    fig.update_xaxes(type="log", title_text="Iteration (logarithmisch)")
    return _base(fig, 340, legend_y=-0.4)


def build_steps(res):
    """Schrittweite eta und Primalgewicht omega je Iteration (log-y)."""
    its = np.arange(1, len(res.eta_path) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=its, y=res.eta_path, mode="lines", line=dict(color=TEAL, width=2), name="Schrittweite η"))
    fig.add_trace(go.Scatter(x=its, y=res.omega_path, mode="lines", line=dict(color=ORANGE, width=2), name="Primalgewicht ω"))
    fig.update_yaxes(type="log", title_text="Wert")
    fig.update_xaxes(title_text="Iteration")
    return _base(fig, 280, legend_y=-0.4)


def build_ablation(rows):
    """Iterationen bis zur Genauigkeit je Konfiguration (Median über die Instanzfamilie, log-x); rot: Läufe, die die Grenze erreichen."""
    colors = {"ref": BLUE, "alone": ORANGE, "removed": TEAL}
    labels = [r["label"] for r in rows][::-1]
    fig = go.Figure(go.Bar(y=labels, x=[max(r["iterations"], 1) for r in rows][::-1], orientation="h", marker_color=[RED if r["optimal"] < r["runs"] else colors[r["kind"]] for r in rows][::-1],
                           text=[f"{r['iterations']:.0f}" + (f"  ({r['optimal']} von {r['runs']} erreichen es)" if r["optimal"] < r["runs"] else "") for r in rows][::-1], textposition="auto"))
    fig.update_xaxes(type="log", title_text="Iterationen (Median)")
    fig.update_yaxes(title_text="")
    return _base(fig, 420)


def build_eps(rows):
    ks = [r["k"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=[r["iterations"] for r in rows], mode="lines+markers", line=dict(color=TEAL, width=3), name="Iterationen (Median)"))
    fig.add_trace(go.Scatter(x=ks, y=[r["matvecs"] for r in rows], mode="lines+markers", line=dict(color=ORANGE, width=2), name="Matrix-Vektor-Produkte (Median)"))
    fails = [r for r in rows if r["optimal"] < r["runs"]]
    if fails:
        fig.add_trace(go.Scatter(x=[r["k"] for r in fails], y=[r["iterations"] for r in fails], mode="markers", marker=dict(size=12, color=RED, symbol="x"), name="nicht alle Läufe erreichen die Genauigkeit"))
    fig.update_xaxes(title_text="Genauigkeit 10^-k", dtick=2)
    fig.update_yaxes(title_text="Anzahl")
    return _base(fig, 300, legend_y=-0.4)


def build_vertex(rows):
    ks = [r["k"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=[r["nnz_x"] for r in rows], mode="lines+markers", line=dict(color=TEAL, width=3), name="Nichtnullen der PDLP-Lösung"))
    fig.add_trace(go.Scatter(x=ks, y=[r["m"] for r in rows], mode="lines", line=dict(color=GREY, width=2, dash="dash"), name="m: höchstens so viele hat eine Ecke"))
    fig.add_trace(go.Scatter(x=ks, y=[r["nnz_vertex"] for r in rows], mode="lines+markers", line=dict(color=BLUE, width=2), name="Nichtnullen der Simplex-Ecke"))
    fig.update_xaxes(title_text="Genauigkeit 10^-k", dtick=2)
    fig.update_yaxes(title_text="Nichtnullen (Standardform)", rangemode="tozero")
    return _base(fig, 300, legend_y=-0.4)


def build_size(sweep):
    rows = sweep["rows"]
    ns = [r["n"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=[r["flops_pdlp"] for r in rows], mode="lines+markers", line=dict(color=ORANGE, width=3), name="PDLP"))
    fig.add_trace(go.Scatter(x=ns, y=[r["flops_ipm"] for r in rows], mode="lines+markers", line=dict(color=PURPLE, width=3), name="Innere Punkte (Mehrotra, dicht)"))
    fig.add_trace(go.Scatter(x=ns, y=[max(r["flops_simplex"], 1) for r in rows], mode="lines+markers", line=dict(color=TEAL, width=3), name="Simplex (dichtes Tableau)"))
    if sweep["cross_ipm"]:
        fig.add_vline(x=sweep["cross_ipm"], line=dict(color=GREY, dash="dot"))
    fig.update_yaxes(type="log", title_text="Operationen (Modell)")
    fig.update_xaxes(title_text="Größe n (Zufall, m = n)", type="log")
    return _base(fig, 340, legend_y=-0.4)


def build_large(rows):
    ns = [r["n"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=[r["flops_pdlp"] for r in rows], mode="lines+markers", line=dict(color=ORANGE, width=3), name="PDLP"))
    fig.add_trace(go.Scatter(x=ns, y=[max(r["flops_simplex"], 1) for r in rows], mode="lines+markers", line=dict(color=TEAL, width=3), name="Simplex (dichtes Tableau)"))
    fig.update_yaxes(type="log", title_text="Operationen (Modell)")
    fig.update_xaxes(title_text="Größe n (Zufall, m = n, Dichte 2 %)", type="log")
    return _base(fig, 300, legend_y=-0.4)


def build_scale(sweep):
    rows, labels = sweep["rows"], sweep["labels"]
    ks = [f"10^{r['k']}" for r in rows]
    colors = [GREY, PURPLE, ORANGE]
    fig = go.Figure()
    for label, color in zip(labels, colors):
        fig.add_trace(go.Bar(x=ks, y=[r[label]["right"] for r in rows], marker_color=color, name=label))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Spalten über 10^k gestreut", type="category")
    fig.update_yaxes(title_text="richtig gelöste Instanzen (von 5)", range=[0, 5.3], fixedrange=True)
    return _base(fig, 320, legend_y=-0.4)


def build_cube(rows):
    ns = [r["n"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=[r["it_pdlp"] for r in rows], mode="lines+markers", line=dict(color=ORANGE, width=3), name="PDLP (Iterationen)"))
    fig.add_trace(go.Scatter(x=ns, y=[r["it_ipm"] for r in rows], mode="lines+markers", line=dict(color=PURPLE, width=3), name="Mehrotra (Iterationen)"))
    fig.add_trace(go.Scatter(x=ns, y=[r["pivots"] for r in rows], mode="lines+markers", line=dict(color=TEAL, width=3), name="Simplex (Pivots, Dantzig)"))
    fig.update_yaxes(type="log", title_text="Iterationen bzw. Pivots")
    fig.update_xaxes(title_text="Größe n des Würfels", dtick=2)
    return _base(fig, 320, legend_y=-0.4)
