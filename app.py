"""PDLP – ein Verfahren erster Ordnung für Lineare Programme - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zehntes Stück der Lineare-Programmierung-Reihe der "Konzepte"-Reihe: Simplex und Innere Punkte zerlegen Matrizen. PDLP verzichtet auf jede Faktorisierung und rechnet nur mit Matrix-Vektor-Produkten:
ein primal-duales Hybrid-Gradienten-Verfahren, das erst mit Vorkonditionierung, Mittelung, Neustarts, adaptiver Schrittweite und Primalgewicht praxistauglich wird. Die Demo baut jedes Stück einzeln ein
und aus und misst, was es bringt.

Lauffähig mit: streamlit run app.py
"""

import math

import pandas as pd
import streamlit as st

import pdl_constants as C
import pdl_evaluation as ev
import pdl_pdlp as P
import pdl_scenario as S
from pdl_evaluation import Settings, analyse
from pdl_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    store_from_widget,
    sync_query_params,
)
from pdl_visualization import (
    build_ablation,
    build_convergence,
    build_cube,
    build_eps,
    build_large,
    build_path,
    build_scale,
    build_size,
    build_steps,
    build_vertex,
)

st.set_page_config(page_title="PDLP – Sebastian Hanisch", layout="wide")


def num(x, digits=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{0.0 if abs(x) < 5e-13 else x:.{digits}f}"


def big(x):
    return f"{x:,.0f}".replace(",", ".")


STATUS_TEXT = {"optimal": "Optimum", "infeasible": "unzulässig (Strahl)", "unbounded": "unbeschränkt (Strahl)", "numerical": "numerisch gescheitert", "limit": "Iterationsgrenze erreicht"}

st.title("🧮 PDLP – Lineare Programme ohne Faktorisierung")
st.markdown(
    """
**Zehntes Stück der Lineare-Programmierung-Reihe.** Der Simplex tauscht Spalten einer Basis, die Inneren Punkte (Stück 8) lösen je Iteration ein lineares Gleichungssystem: beide zerlegen Matrizen, und das wird bei riesigen dünnen LPs teuer.
**PDLP** (Applegate u. a. 2021) rechnet nur mit **Matrix-Vektor-Produkten**: ein primal-duales Hybrid-Gradienten-Verfahren (PDHG) auf dem Sattelpunkt des LPs, je Iteration zwei Produkte, kein Faktorisieren. Die Grundform
konvergiert langsam; erst **fünf Bausteine** (Vorkonditionierung, Mittelung, Neustarts, adaptive Schrittweite, Primalgewicht) machen daraus ein Verfahren, das HiGHS, COPT, Xpress und cuOpt anbieten. Fünf Fragen, alle gemessen:
**(1) Der Weg** - wie sieht PDHG aus? **(2) Konvergenz** - wie schnell fallen die Residuen? **(3) Die fünf Bausteine** - was bringt jeder? **(4) Gegen Simplex und Innere Punkte** und **(5) Grenzen** - Genauigkeit, keine Ecke, Skalierung.
"""
)
st.caption("Kind der [Inneren Punkte](https://github.com/sebastian-hanisch/innere-punkte-demo); greift die Skalierungsbefunde aus [Präsolve und Numerik](https://github.com/sebastian-hanisch/praesolve-demo) auf. Crossover ist gebaut: [crossover-demo](https://github.com/sebastian-hanisch/crossover-demo).")

with st.expander("So funktioniert PDLP", expanded=True):
    st.markdown(
        """
1. **Sattelpunkt:** min c·x unter M x = b, x ≥ 0 ist gleichwertig zu min über x ≥ 0 und max über y von c·x − y·(M x − b). **PDHG** (Chambolle & Pock 2011) macht abwechselnd einen Gradientenschritt in x (mit Projektion auf x ≥ 0) und einen in y,
   mit der Extrapolation 2x⁺ − x im dualen Schritt. Je Iteration ein Produkt M x und eins Mᵀ y; die Schrittweite muss unter 1 / ‖M‖ bleiben.
2. **Mittelung und Neustarts:** der Mittelwert der Iterierten konvergiert garantiert (sublinear), die Iterierte selbst zickzackt. Startet man immer wieder vom besseren der beiden neu, wird die Konvergenz **linear** (Applegate u. a. 2021).
3. **Vorkonditionierung:** Zeilen und Spalten werden skaliert (Ruiz, Pock-Chambolle), damit die Matrix gut konditioniert ist; ohne sie ist das Verfahren auf Daten in verschiedenen Einheiten praktisch unbrauchbar (Bezug: Stück 9).
4. **Adaptive Schrittweite und Primalgewicht:** die Schrittweite wächst, solange eine lokale Bedingung hält, und wird sonst verkleinert; das Primalgewicht ω stimmt die Schritte in x und y aufeinander ab.
5. **Ergebnis:** eine **Näherung im Inneren einer optimalen Fläche**, keine Ecke; Duale kommen mit (y). Das Verfahren stoppt, wenn primales Residuum, duales Residuum und Lücke unter ε liegen - am Original nachgerechnet.
        """
    )

if C.PRESETS:
    st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
    preset_names = list(C.PRESETS.keys())
    for row in (preset_names[:4], preset_names[4:8], preset_names[8:]):
        if not row:
            continue
        cols = st.columns(len(row))
        for col, name in zip(cols, row):
            with col:
                st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

ss = st.session_state
with st.sidebar:
    st.header("⚙️ Einstellungen")
    kind = st.selectbox("Instanz", options=list(S.KINDS), format_func=lambda v: S.KIND_LABELS[v], key="kind_select",
                        help="Lehrbuchbeispiel, Zentrum und entartete Ecke sind fest; Zufall, Mischung (mit ≥ und =) und Plateau (eine ganze optimale Fläche) sind regelbar; der Klee-Minty-Würfel hat nur die Größe n. Unzulässig und Unbeschränkt zeigen die Strahlen.")
    cube = kind in S.CUBE_KINDS
    random_kind = kind in ev.RANDOM_KINDS
    if random_kind:
        m = st.slider("Ressourcen m", *bounds("m_slider"), value=int(ss["m_slider"]), key="m_widget", on_change=store_from_widget, args=("m_slider",), help="Zahl der Bedingungen.")
    else:
        m = C.DEFAULT_M
    if random_kind or cube:
        cap_n = S.CUBE_MAX if cube else C.N_MAX
        n = st.slider("Dienste n", C.N_MIN, cap_n, value=min(int(ss["n_slider"]), cap_n), key="n_widget", on_change=store_from_widget, args=("n_slider",), help="Zahl der Variablen (beim Würfel auch die Zahl der Ressourcen).")
    else:
        n = C.DEFAULT_N
    if random_kind:
        density_i = st.select_slider("Dichte der Matrix", options=list(range(len(C.DENSITIES))), value=int(ss["density_select"]), format_func=C.density_label, key="density_widget", on_change=store_from_widget,
                                     args=("density_select",), help="Anteil der Nichtnullen; PDLP kostet je Produkt nur so viele Operationen wie die Matrix Nichtnullen hat.")
        seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), value=int(ss["seed_input"]), key="seed_widget", step=1, on_change=store_from_widget, args=("seed_input",))
        st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)
    else:
        density_i, seed = C.DEFAULT_DENSITY_I, C.DEFAULT_SEED
    eps_i = st.select_slider("Genauigkeit ε", options=list(range(len(C.EPS_EXPS))), format_func=C.eps_label, key="eps_select", help="Relative Residuen und Lücke, bei denen das Verfahren stoppt (am Original nachgerechnet).")
    cap_i = st.select_slider("Iterationsgrenze", options=list(range(len(C.CAPS))), format_func=C.cap_label, key="cap_select", help="Nach so vielen Iterationen bricht das Verfahren ab; ein Lauf an der Grenze ist kein Ergebnis.")
    scale_i = st.select_slider("Skalierung der Spalten", options=list(range(len(C.SCALE_EXPS))), format_func=C.scale_label, key="scale_select",
                               help="Die Dienste werden mit Faktoren zwischen 1 und 10^k umgerechnet (Optimalwert gleich, Zahlen sehr verschieden groß): ein Test für die Vorkonditionierung.")
    st.markdown("**Bausteine von PDLP**")
    blocks = {}
    for block, key in C.BLOCK_KEYS.items():
        blocks[block] = st.toggle(P.BLOCK_LABELS[block], key=key, help=C.BLOCK_HELP[block])

sync_query_params({"kind_select": kind, "m_slider": int(ss["m_slider"]), "n_slider": int(ss["n_slider"]), "seed_input": int(ss["seed_input"]), "density_select": int(ss["density_select"]), "eps_select": int(eps_i),
                   "cap_select": int(cap_i), "scale_select": int(scale_i), **{key: bool(blocks[b]) for b, key in C.BLOCK_KEYS.items()}, "pdl_step": int(ss["pdl_step"])})

settings = Settings(kind, int(m), int(n), int(seed), int(density_i), int(eps_i), int(cap_i), int(scale_i), blocks["average"], blocks["restart"], blocks["precond"], blocks["adaptive"], blocks["primal_weight"])
with st.spinner("Rechne..."):
    a = analyse(settings)
res, inst = a.res, a.inst
cfg_txt = "alle fünf Bausteine" if settings.blocks == ev.FULL else ("Grundform" if settings.blocks == ev.BASIC else ", ".join(P.BLOCK_LABELS[b] for b in P.BLOCKS if settings.blocks[b]))

# --- In Aktion ---------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Von der Grundform zu PDLP")
step = st.select_slider("Schritt", options=list(C.STEPS), key="pdl_step", format_func=lambda s: C.STEPS[s])

if res.status == "optimal":
    if math.isfinite(a.error) and a.error > 1e-3:
        st.warning(f"⚠️ Das Verfahren meldet ein Optimum ({num(res.obj)}), aber der Optimalwert weicht um {a.error:.1%} vom Referenzwert ({num(a.ref_obj)}) ab.")
    else:
        st.success(f"✅ Optimum **{num(res.obj)}** nach **{res.iterations}** Iterationen und {res.matvecs} Matrix-Vektor-Produkten ({cfg_txt}, ε = {settings.eps:g}); Simplex: {num(a.ref_obj)} nach {a.simplex.pivots} Pivots.")
elif res.status in ("infeasible", "unbounded"):
    st.info(f"Die Instanz ist **{'unzulässig' if res.status == 'infeasible' else 'unbeschränkt'}**: {res.note} (nach {res.iterations} Iterationen). Der Strahl ist nachgerechnet und ein Beweis, kein Verdacht.")
elif res.status == "numerical":
    st.error(f"❌ Numerisch gescheitert nach {res.iterations} Iterationen: {res.note}.")
else:
    st.warning(f"Iterationsgrenze ({big(settings.cap)}) erreicht ohne Ergebnis: {res.note}. Der Simplex meldet: {a.simplex.status}. Kein Optimalitätsanspruch; mit Neustarts und Vorkonditionierung oder einer höheren Grenze käme man weiter.")

if step == 1:
    total = res.iterations
    if inst.n == 2 and len(res.points) > 1:
        ss["iter_k"] = min(max(0, int(ss.get("iter_k", total))), total)
        k = st.slider("Iteration", 0, total, key="iter_k", help="0 = Startpunkt (0, 0); danach Schritt für Schritt.") if total > 0 else 0
        idx = min(k // P.CHECK_EVERY, len(res.check_iters)) - 1
        eta_k = res.eta_path[k - 1] if k > 0 else float("nan")
        st.markdown(f"**Iteration {k} von {total}:** Schrittweite η = {num(eta_k, 3)}, Primalgewicht ω = {num(res.omega_path[k - 1], 3) if k > 0 else '-'}"
                    + (f"; Residuen des Kandidaten: primal {res.prim_res[idx]:.1e}, dual {res.dual_res[idx]:.1e}, Lücke {res.gap[idx]:.1e}." if idx >= 0 else "."))
        st.plotly_chart(build_path(a, k), width="stretch", key=f"s1_path_{k}")
        st.caption("Orange: die Iterierten. Sie liegen fast immer außerhalb der zulässigen Menge, denn PDHG erzwingt nur x ≥ 0 und erfüllt die Nebenbedingungen erst im Grenzwert; lila gepunktet: der Mittelwert; blaue Rauten: Neustarts; rot: die aktuelle Iterierte; Stern: Optimum des Simplex.")
    else:
        st.markdown("Ab drei Diensten gibt es kein Bild des Wegs; die Kurven in Schritt 2 zeigen ihn: die Residuen fallen, mit Neustarts fast geradlinig.")
elif step == 2:
    if res.check_iters:
        st.plotly_chart(build_convergence(res, a.basic), width="stretch", key="s2_conv")
        st.caption("Beide Achsen logarithmisch; ausgewertet wird alle vier Iterationen. Ohne Neustarts fallen die Residuen langsam und ruckelnd (sublinear); mit Neustarts geradlinig (linear), die orangen Rauten sind die Neustarts. Grau gestrichelt: die Grundform derselben Instanz.")
    if res.eta_path:
        st.plotly_chart(build_steps(res), width="stretch", key="s2_steps")
        st.caption("Schrittweite η (mit adaptiver Regel wächst sie, bis die Bedingung anschlägt) und Primalgewicht ω (springt bei jedem Neustart).")
    c1, c2, c3 = st.columns(3)
    c1.metric("Iterationen", big(res.iterations), delta=f"Grundform: {big(a.basic.iterations)}", delta_color="off")
    c2.metric("Produkte", big(res.matvecs), delta=f"Grundform: {big(a.basic.matvecs)}", delta_color="off")
    c3.metric("Neustarts", str(res.restarts), delta=f"{res.rejected} abgelehnte Schrittweiten", delta_color="off")
elif step == 3:
    st.markdown("**Was bringt jeder Baustein?** Zwölf Konfigurationen auf der gewählten Instanz (Zufall: Median über fünf feste Instanzen); 🔬 auf Abruf, etwa 10 Sekunden.")
    tok = (settings.kind, settings.m, settings.n, settings.seed, settings.density_i, settings.eps_i, settings.scale_i, min(settings.cap, ev.ABLATION_CAP))
    if st.button("Ablation der fünf Bausteine berechnen", key="ablation_start"):
        ss["ablation_done"] = tok
    if ss.get("ablation_done") == tok:
        with st.spinner("Rechne..."):
            rows = ev.ablation(settings)
        st.plotly_chart(build_ablation(rows), width="stretch", key="s3_ablation")
        st.dataframe(pd.DataFrame([{"Konfiguration": r["label"], "Iterationen (Median)": big(r["iterations"]), "Produkte (Median)": big(r["matvecs"]), "Läufe mit Ergebnis": f"{r['optimal']} von {r['runs']}"} for r in rows]), hide_index=True, width="stretch")
        st.caption(f"Grenze {big(rows[0]['cap'])} Iterationen. Blau: Grundform und alle fünf; orange: ein Baustein allein zur Grundform; grün: alle fünf ohne einen; rot: mindestens ein Lauf erreicht die Genauigkeit nicht. "
                   "Mittelung oder Neustarts allein ändern nichts: erst zusammen wirken sie; die Vorkonditionierung ist allein der stärkste Baustein.")
    st.markdown("**Auf dieser Instanz** (gewählte Bausteine gegen Grundform und alle fünf):")
    st.dataframe(pd.DataFrame([{"Konfiguration": lab, "Ergebnis": STATUS_TEXT[r.status], "Iterationen": big(r.iterations), "Produkte": big(r.matvecs), "Neustarts": r.restarts} for lab, r in
                               (("Grundform", a.basic), (f"gewählt ({cfg_txt})", res), ("alle fünf", a.full))]), hide_index=True, width="stretch")
elif step == 4:
    st.markdown("**PDLP gegen Simplex und Innere Punkte auf dieser Instanz** (Operationsmodell, kein Wandzeit-Vergleich):")
    ip = a.ipm
    st.dataframe(pd.DataFrame([
        {"Verfahren": "PDLP (gewählte Bausteine)", "Ergebnis": STATUS_TEXT[res.status], "Iterationen / Pivots": big(res.iterations), "Operationen (Modell)": big(a.flops_pdlp)},
        {"Verfahren": "Innere Punkte (Mehrotra, dichte Cholesky)", "Ergebnis": STATUS_TEXT.get(ip.status, ip.status), "Iterationen / Pivots": big(ip.iterations), "Operationen (Modell)": big(a.flops_ipm)},
        {"Verfahren": "Simplex (dichtes Tableau)", "Ergebnis": a.simplex.status, "Iterationen / Pivots": big(a.simplex.pivots), "Operationen (Modell)": big(a.flops_simplex)},
    ]), hide_index=True, width="stretch")
    if a.flops_simplex and res.status == "optimal":
        st.markdown(f"PDLP braucht auf dieser Instanz das **{a.flops_pdlp / a.flops_simplex:.1f}-Fache** der Operationen des dichten Tableau-Simplex" + (f" und das {a.flops_pdlp / a.flops_ipm:.2f}-Fache der Inneren Punkte." if a.flops_ipm else "."))
    st.caption("Modell: 2·nnz Operationen je Matrix-Vektor-Produkt (nur die Nichtnullen zählen), dazu etwa 16 Vektoroperationen je Iteration; Innere Punkte: 2m²N + m³/3 + … je Iteration (dicht); Simplex: 2(m+1)(Spalten+1) je Pivot (dicht). Echte Löser rechnen dünn: das verkleinert die Modellkosten von Simplex und Innerem Punkt.")
    tok = (settings.density_i, settings.eps_i)
    if st.button("Operationen über die Größe berechnen", key="size_start"):
        ss["size_done"] = tok
    if ss.get("size_done") == tok:
        with st.spinner("Rechne..."):
            sweep = ev.size_sweep(settings)
        st.plotly_chart(build_size(sweep), width="stretch", key="s4_size")
        st.dataframe(pd.DataFrame([{"n": r["n"], "Iterationen PDLP": big(r["it_pdlp"]), "Iterationen Mehrotra": big(r["it_ipm"]), "Pivots": big(r["pivots"]), "PDLP / Innere Punkte": f"{r['ratio_ipm']:.2f}", "PDLP / Simplex": f"{r['ratio_simplex']:.1f}"}
                                    for r in sweep["rows"]]), hide_index=True, width="stretch")
        st.caption(f"Zufallsinstanzen m = n, Dichte {settings.density:.0%}, Median über drei feste Instanzen. "
                   + (f"Ab n = {sweep['cross_ipm']} braucht PDLP weniger Operationen als die Inneren Punkte (dichtes Modell; senkrechte Linie). " if sweep["cross_ipm"] else "PDLP braucht in diesem Bereich immer mehr Operationen als die Inneren Punkte. ")
                   + (f"Den Simplex schlägt es ab n = {sweep['cross_simplex']}." if sweep["cross_simplex"] else "Den Simplex schlägt es in diesem Bereich nie."))
    tok_l = (settings.eps_i,)
    if st.button("Große dünne Instanzen (n bis 500)", key="large_start"):
        ss["large_done"] = tok_l
    if ss.get("large_done") == tok_l:
        with st.spinner("Rechne..."):
            rows = ev.large_sparse(settings)
        st.plotly_chart(build_large(rows), width="stretch", key="s4_large")
        st.dataframe(pd.DataFrame([{"n": r["n"], "Nichtnullen": big(r["nnz"]), "Iterationen PDLP": big(r["iterations"]), "Pivots": big(r["pivots"]), "PDLP / Simplex": f"{r['ratio']:.1f}"} for r in rows]), hide_index=True, width="stretch")
        st.caption("Dichte 2 %, je eine feste Instanz; Mehrotra entfällt (die dichte Zerlegung braucht hier Minuten). Der Simplex braucht auf diesen Instanzen nur wenige Pivots: der Kreuzungspunkt liegt jenseits dessen, was die Demo zeigt.")
    st.markdown("**Der schlimmste Fall des Simplex:** der Klee-Minty-Würfel aus Stück 3 (🔬 auf Abruf).")
    tok_c = (settings.eps_i,)
    if st.button("Auf dem Würfel vergleichen", key="cube_start"):
        ss["cube_done"] = tok_c
    if ss.get("cube_done") == tok_c:
        with st.spinner("Rechne..."):
            rows = ev.cube_sweep(settings)
        st.plotly_chart(build_cube(rows), width="stretch", key="s4_cube")
        st.dataframe(pd.DataFrame([{"n": r["n"], "PDLP-Iterationen": r["it_pdlp"], "Mehrotra-Iterationen": r["it_ipm"], "Pivots (Dantzig)": big(r["pivots"])} for r in rows]), hide_index=True, width="stretch")
        st.caption("Der Simplex besucht alle 2^n Ecken; PDLP braucht für alle n etwa gleich viele Iterationen, die Inneren Punkte nur wenige mehr mit n.")
else:
    st.markdown("**Wie viele Iterationen kostet jede Stelle?** Und ist das Ergebnis eine Ecke? (🔬 auf Abruf; gewählte Bausteine, Zufall: Median über fünf feste Instanzen)")
    tok_e = (settings.kind, settings.m, settings.n, settings.seed, settings.density_i, settings.cap_i, settings.scale_i, tuple(settings.blocks.values()))
    if st.button("Genauigkeit verschärfen", key="eps_start"):
        ss["eps_done"] = tok_e
    if ss.get("eps_done") == tok_e:
        with st.spinner("Rechne..."):
            rows = ev.eps_sweep(settings)
        st.plotly_chart(build_eps(rows), width="stretch", key="s5_eps")
        st.plotly_chart(build_vertex(rows), width="stretch", key="s5_vertex")
        st.dataframe(pd.DataFrame([{"ε": f"10^-{r['k']}", "Iterationen": big(r["iterations"]), "Produkte": big(r["matvecs"]), "Nichtnullen PDLP": f"{r['nnz_x']:.0f}" if math.isfinite(r["nnz_x"]) else "-",
                                     "Nichtnullen Ecke": f"{r['nnz_vertex']:.0f}" if math.isfinite(r["nnz_vertex"]) else "-", "m": r["m"], "Läufe mit Ergebnis": f"{r['optimal']} von {r['runs']}"} for r in rows]), hide_index=True, width="stretch")
        st.caption("Nichtnullen: Komponenten der Standardform (Dienste, Schlupf, Überschuss) über 10^-6 der größten. Eine Ecke hat höchstens m davon. Auf Zufallsinstanzen mit eindeutigem Optimum konvergiert PDLP gegen genau diese Ecke; "
                   "auf dem Plateau (Instanz wählen) landet es im Inneren der optimalen Fläche: fast alle Komponenten sind positiv. Das ist der Anlass für **Crossover**.")
    tok_s = (settings.density_i, settings.cap_i)
    if st.button("Schlechte Skalierung testen (etwa 30 s)", key="scale_start"):
        ss["scale_done"] = tok_s
    if ss.get("scale_done") == tok_s:
        with st.spinner("Rechne..."):
            sweep = ev.scale_sweep(settings)
        st.plotly_chart(build_scale(sweep), width="stretch", key="s5_scale")
        st.dataframe(pd.DataFrame([{"Spalten über": f"10^{r['k']}", **{lab: f"{r[lab]['right']} richtig" + (f", {r[lab]['iterations']:.0f} Iterationen" if r[lab]["right"] else "") for lab in sweep["labels"]}} for r in sweep["rows"]]), hide_index=True, width="stretch")
        st.caption("Fünf feste 10 × 10-Instanzen, Spalten mit Faktoren zwischen 1 und 10^k umgerechnet (Optimalwert gleich), ε = 10^-6. 'Richtig': Optimum auf 0.1 % genau. Ohne Vorkonditionierung scheitert PDLP schon ab 10^2: viel früher als der Simplex (Stück 9).")
    tok_d = (settings.cap_i,)
    if st.button("Unzulässig und unbeschränkt erkennen", key="detect_start"):
        ss["detect_done"] = tok_d
    if ss.get("detect_done") == tok_d:
        with st.spinner("Rechne..."):
            det = ev.detection(settings)
        st.dataframe(pd.DataFrame([{"Instanzen": lab, "Strahl gefunden (Beweis)": f"{det[key]['found']} von {det[key]['runs']}", "Iterationsgrenze": det[key]["limit"], "Iterationen (Median)": big(det[key]["iterations"]) if det[key]["found"] else "-"}
                                    for lab, key in (("30 konstruierte unzulässige", "infeasible"), ("30 konstruierte unbeschränkte", "unbounded"))]), hide_index=True, width="stretch")
        st.caption("Ein Optimum wird nie gemeldet. Ein Strahl (Farkas bzw. Unbeschränktheit) wird alle 64 Iterationen nachgerechnet; ohne Strahl bleibt es bei der Iterationsgrenze, ein Verdacht ohne Beweis.")

st.markdown("---")
st.markdown("## ⚙️ Der gewählte Fall")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Iterationen", str(res.iterations), delta=STATUS_TEXT[res.status], delta_color="off")
m2.metric("Produkte / Neustarts", f"{res.matvecs} / {res.restarts}", delta="Matrix-Vektor / Neustarts", delta_color="off")
m3.metric("PDLP / Simplex", f"{a.flops_pdlp / a.flops_simplex:,.1f}×".replace(",", ".") if a.flops_simplex and a.flops_pdlp else "-", delta="Operationen (Modell)", delta_color="off")
m4.metric("Ergebnis", num(res.obj) if res.x else "-", delta=("Referenz " + num(a.ref_obj)) if math.isfinite(a.ref_obj) else "keine Referenz", delta_color="off")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **PDLP ist die Antwort auf große LPs.** | Auf allen Größen, die die Demo zeigt (bis n = 500 dünn), braucht PDLP im Operationsmodell mehr als der Simplex; nur gegen die dichten Inneren Punkte gewinnt es ab n ≈ 64. Der Vorteil liegt bei Matrizen, die man weder faktorisieren noch im Speicher als Tableau halten kann (Millionen von Nichtnullen, GPU). | Echte Großinstanzen, GPU |
| **Die Grundform genügt.** | Die Grundform von PDHG erreicht auf den Zufallsinstanzen selbst 10^-4 oft nur nach über zehntausend Iterationen, bei 10^-6 meist nicht; die Vorkonditionierung ist der stärkste einzelne Baustein, Mittelung und Neustarts wirken nur zusammen. | Die fünf Bausteine |
| **Skalierung ist ein Randthema.** | Ohne Vorkonditionierung scheitert PDLP schon bei Spalten über 10^2; mit ihr bleibt es bis 10^8 zuverlässig, bei 10^12 lösen nur noch 3 von 5 Instanzen. | Präsolve und Skalierung (Stück 9) |
| **Das Ergebnis ist eine Ecke.** | Bei eindeutigem Optimum ja, näherungsweise; bei einer optimalen Fläche (Plateau) landet PDLP im Inneren. Basis, Duale mit Ranging und Warmstart bekommt man erst mit **Crossover**. | Crossover |
| **Ein Verdacht ist ein Beweis.** | Unzulässigkeit und Unbeschränktheit gelten nur mit nachgerechnetem Strahl; sonst meldet die Demo die Iterationsgrenze, ohne Optimalitätsanspruch. Die Demo hat keine Feasibility-Polishing-Stufe. | Feasibility Polishing |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Standardform:** $\min c^\top x$ unter $Mx = b$, $x \ge 0$; Sattelpunkt $\min_{x\ge 0}\max_y\; c^\top x - y^\top(Mx - b)$. **PDHG-Schritt** ($\tau = \eta/\omega$, $\sigma = \eta\omega$):
$x^+ = \max(0,\, x - \tau(c - M^\top y))$, $y^+ = y + \sigma(b - M(2x^+ - x))$; Konvergenz für $\eta \le 1/\|M\|_2$. **Adaptive Schrittweite:** $\eta' = \min\big((1-(k+1)^{-0.3})\,\|\Delta z\|_\omega^2 / (2|\Delta y^\top M \Delta x|),\; (1+(k+1)^{-0.6})\,\eta\big)$ mit
$\|z\|_\omega^2 = \omega\|x\|^2 + \|y\|^2/\omega$; verworfen wird, solange $\eta > \eta'$. **KKT-Fehler:** $\sqrt{\omega^2\|Mx-b\|^2 + \|(M^\top y - c)^+\|^2/\omega^2 + (c^\top x - b^\top y)^2}$; **Neustart** auf den besseren von Iterierter und Mittelwert, wenn dieser Fehler auf 0.2 des Fehlers am
letzten Neustart fällt (oder auf 0.8 ohne Fortschritt, oder künstlich nach 0.36 der Iterationen). **Primalgewicht:** $\omega \leftarrow \exp(0.5\ln(\Delta y/\Delta x) + 0.5\ln\omega)$. **Abbruch:** relative Residuen $\|Mx-b\|/(1+\|b\|)$, $\|(M^\top y-c)^+\|/(1+\|c\|)$ und $|c^\top x - b^\top y|/(1+|c^\top x|+|b^\top y|)$ jeweils $\le\varepsilon$, am Original.

**Literatur.** Applegate, D., Díaz, M., Hinder, O., Lu, H., Lubin, M., O'Donoghue, B., & Schudy, W. (2021). *Practical large-scale linear programming using primal-dual hybrid gradient.* Advances in Neural Information Processing Systems 34.
Chambolle, A., & Pock, T. (2011). *A first-order primal-dual algorithm for convex problems with applications to imaging.* Journal of Mathematical Imaging and Vision 40(1), 120-145. Lu, H., & Yang, J. (2023). *cuPDLP.jl: a GPU implementation of restarted primal-dual hybrid gradient for linear programming in Julia* (arXiv 2311.12180; hier nur genannt).

Implementiert in `pdl_pdlp.py` (PDHG, fünf Bausteine, Strahltests), `pdl_algorithm.py` (Tableau-Simplex) und `pdl_ipm.py` (Mehrotra) als Vergleich, `pdl_evaluation.py`, `pdl_scenario.py`.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Lineare Programmierung: vom Tableau zum Crossover](https://sebastianhanisch.net/konzepte-lineare-programmierung.html)."
)
