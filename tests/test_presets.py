"""Presets: gültige Werte und jede Zahl der Hilfetexte gegen die echten Auswertungsfunktionen (Iterationszahlen und Numerik-Grenzfälle nur als Bänder: Windows und Linux können abweichen)."""

import numpy as np
import pytest

import pdl_constants as C
import pdl_evaluation as ev
import pdl_pdlp as P
from pdl_presets import PRESET_KEYS, SETTING_SPECS
from tests.test_scenario import _highs


def _settings(name, **over):
    p = {**C.PRESETS[name], **over}
    return ev.Settings(p["kind"], p["m"], p["n"], p["seed"], p["density"], p["eps"], p["cap"], p["scale"], p["average"], p["restart"], p["precond"], p["adaptive"], p["primal_weight"])


def _has(name, *values):
    for v in values:
        assert v in C.PRESET_HELP[name], (name, v)


def _outside(res):
    """Anteil der Iterierten des Lehrbuchbeispiels außerhalb der zulässigen Menge x1 <= 4, 2 x2 <= 12, 3 x1 + 2 x2 <= 18."""
    pts = np.array(res.points)
    viol = np.maximum.reduce([pts[:, 0] - 4, 2 * pts[:, 1] - 12, 3 * pts[:, 0] + 2 * pts[:, 1] - 18])
    return int((viol > 1e-6).sum()), len(pts)


def test_every_preset_has_valid_values_and_a_help_text():
    assert list(C.PRESETS) == list(C.PRESET_HELP) and len(C.PRESETS) == 12
    for name, p in C.PRESETS.items():
        assert set(p) <= set(PRESET_KEYS) and {"kind", "step", "density", "eps", "cap", "scale", "average", "restart", "precond", "adaptive", "primal_weight"} <= set(p), name
        for key, state_key in PRESET_KEYS.items():
            if key in p and state_key in SETTING_SPECS:
                spec = SETTING_SPECS[state_key]
                assert spec.caster(str(int(p[key])) if isinstance(p[key], bool) else p[key]) == p[key], (name, key)
                if spec.lo is not None:
                    assert spec.lo <= p[key] <= spec.hi, (name, key)
        assert C.PRESET_HELP[name].strip()


def test_help_lehrbuch_presets():
    name = "Lehrbuch: der Weg der Grundform"
    a = ev.analyse(_settings(name))
    r = a.res
    assert r.status == "optimal" and 300 <= r.iterations <= 380 and r.matvecs == 2 * P.NORM_ITERS + 2 * r.iterations and a.simplex.pivots == 2 and a.ref_obj == pytest.approx(36.0)
    out, total = _outside(r)
    assert total == r.iterations + 1 and out >= 0.9 * total
    _has(name, "36 bei (2, 6)", "340 Iterationen", "760 Matrix-Vektor-Produkte", "2 Pivots", "Iteration 200", "334 von 341")
    name = "Lehrbuch: PDLP mit Neustarts"
    a = ev.analyse(_settings(name))
    r = a.res
    assert r.status == "optimal" and 50 <= r.iterations <= 100 and 5 <= r.restarts <= 12 and 10 <= r.rejected <= 45 and a.basic.iterations >= 3 * r.iterations
    assert np.array(r.points)[:, 1].max() > 10 and len(r.avg_points) == len(r.points)
    _has(name, "72 Iterationen", "249 Produkte", "340 und 760", "8 Neustarts", "25 abgelehnte", "(7.7, 16.9)")


def test_help_grundform_and_scaling_presets():
    name = "Die Grundform ist langsam"
    a = ev.analyse(_settings(name))
    assert a.res.status == "optimal" and 9000 <= a.res.iterations <= 16000 and 100 <= a.full.iterations <= 250 and a.res.iterations >= 40 * a.full.iterations and a.simplex.pivots == 3
    _has(name, "12 × 12", "12.888 Iterationen", "25.856 Produkte", "152 Iterationen", "426 Produkte", "85-mal", "3 Pivots")
    name = "Vorkonditionierung rettet die Spanne"
    a = ev.analyse(_settings(name))
    assert a.res.status == "optimal" and 300 <= a.res.iterations <= 900 and 10 <= a.res.restarts <= 30 and a.error < 1e-4 and a.ref_obj == pytest.approx(285.544, abs=1e-3)
    _has(name, "10 × 10", "10^6", "500 Iterationen", "18 Neustarts", "285.54")
    name = "Ohne Vorkonditionierung hängt es"
    a = ev.analyse(_settings(name))
    assert a.res.status == "limit" and a.res.iterations == 20000 and 0.05 < a.error < 0.3 and "Iterationsgrenze" in a.res.note and a.full.status == "optimal"
    _has(name, "20.000 Iterationen", "14 %", "244.58 statt 285.54", "nie als Optimum")


def test_help_ablation_and_ladder_presets():
    name = "Die fünf Bausteine im Vergleich"
    s = _settings(name)
    a = ev.analyse(s)
    assert a.res.status == "optimal" and 250 <= a.res.iterations <= 500 and a.basic.status == "limit" and a.basic.iterations == 20000
    rows = {r["label"]: r for r in ev.ablation(s)}
    assert len(rows) == 12 and 200 <= rows["PDLP (alle fünf)"]["iterations"] <= 400
    assert 2000 <= rows["+ Vorkonditionierung"]["iterations"] <= 5000 and rows["+ Vorkonditionierung"]["optimal"] == 5
    assert 8000 <= rows["+ adaptive Schrittweite"]["iterations"] <= 20000
    assert all(rows[k]["iterations"] == 20000 and rows[k]["optimal"] <= 2 for k in ("+ Mittelung", "+ Neustarts", "+ Primalgewicht", "Grundform")) and rows["Grundform"]["optimal"] == 0
    assert all(400 <= rows[f"ohne {P.BLOCK_LABELS[b]}"]["iterations"] <= 2500 for b in P.BLOCKS)
    _has(name, "20 × 20", "356", "3.228", "11.104", "284", "564 bis 1.736")
    name = "Jede Stelle kostet wenig"
    rows = {r["k"]: r for r in ev.eps_sweep(_settings(name))}
    assert 40 <= rows[2]["iterations"] <= 120 and 200 <= rows[4]["iterations"] <= 400 and 500 <= rows[6]["iterations"] <= 850 and 750 <= rows[8]["iterations"] <= 1200 and 900 <= rows[10]["iterations"] <= 1500
    assert all(rows[k]["optimal"] == 5 for k in rows) and all(rows[b]["iterations"] - rows[a]["iterations"] < 400 for a, b in ((2, 4), (4, 6), (6, 8), (8, 10)))
    _has(name, "72 Iterationen", "284", "652", "948", "1.124", "unter 200")


def test_help_vertex_and_size_presets():
    name = "Plateau: keine Ecke"
    s = _settings(name)
    rows = {r["k"]: r for r in ev.eps_sweep(s)}
    assert all(rows[k]["nnz_vertex"] == 10 and rows[k]["m"] == 10 and rows[k]["nnz_x"] >= 19 and rows[k]["optimal"] == 5 for k in rows)
    _has(name, "10 × 12", "10 Nichtnullen", "21 bis 22 von 22", "10^-10")
    name = "Gegen Innere Punkte: dünn und groß"
    a = ev.analyse(_settings(name))
    assert a.res.status == "optimal" and 150 <= a.res.iterations <= 400 and 4 <= a.ipm.iterations <= 10 and 0.2 < a.flops_pdlp / a.flops_ipm < 0.5 and 10 < a.flops_pdlp / a.flops_simplex < 40 and a.simplex.pivots == 5
    _has(name, "40 × 40", "5 % Dichte", "236 Iterationen", "653.404", "7 Iterationen", "2.170.931", "3.3-mal", "5 Pivots", "33.210", "19.7-mal")
    name = "Klee-Minty-Würfel"
    a = ev.analyse(_settings(name))
    assert a.res.status == "optimal" and 30 <= a.res.iterations <= 90 and a.simplex.pivots == 2 ** 14 - 1 and 10 <= a.ipm.iterations <= 20 and a.flops_pdlp < a.flops_ipm < a.flops_simplex
    _has(name, "n = 14", "16.383 Pivots", "15 Iterationen", "44", "14.253.210", "241.350", "75.978")


def test_help_ray_presets():
    name = "Unzulässig: Farkas-Strahl"
    a = ev.analyse(_settings(name))
    assert a.res.status == "infeasible" and 100 <= a.res.iterations <= 400 and "Farkas" in a.res.note and _highs(a.inst).status == 2
    _has(name, "192 Iterationen", "x1 ≥ 6", "x1 ≤ 4", "Beweis")
    name = "Unbeschränkt: Strahl"
    a = ev.analyse(_settings(name))
    assert a.res.status == "unbounded" and 300 <= a.res.iterations <= 1500 and "unbeschränkt" in a.res.note
    _has(name, "704 Iterationen", "Mx = 0", "Beweis")
