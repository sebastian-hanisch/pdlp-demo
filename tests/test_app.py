"""AppTest-Rauchtests: Voreinstellung, jedes Preset, jeder Schritt für jede Instanz, jeder Baustein-Schalter, Regler-Randwerte, Permalink-Grenzen, bedingte Regler, Berechnungen auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import pdl_constants as C
import pdl_pdlp as P
import pdl_scenario as S

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(step=2, **state):
    at = AppTest.from_file(APP, default_timeout=300)
    state.setdefault("pdl_step", step)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m for m in at.metric if m.label == label)


def _click(at, key):
    next(b for b in at.button if b.key == key).click().run()


def test_default_run_shows_the_full_pdlp_result():
    at = _run()
    _ok(at)
    assert {"Iterationen", "Produkte / Neustarts", "PDLP / Simplex", "Ergebnis"} <= {m.label for m in at.metric}
    assert any("Optimum" in s.value and "alle fünf Bausteine" in s.value for s in at.success) and at.get("plotly_chart")
    assert len(at.toggle) == 5 and all(t.value for t in at.toggle)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run(kind_select="centre")
    _click(at, f"preset_{name}")
    _ok(at)
    p = C.PRESETS[name]
    ss = at.session_state
    assert (ss["kind_select"], ss["density_select"], ss["eps_select"], ss["cap_select"], ss["scale_select"], ss["average_toggle"], ss["restart_toggle"], ss["precond_toggle"], ss["adaptive_toggle"], ss["pw_toggle"], ss["pdl_step"]) == (
        p["kind"], p["density"], p["eps"], p["cap"], p["scale"], p["average"], p["restart"], p["precond"], p["adaptive"], p["primal_weight"], p["step"])
    if "iter_k" in p:
        assert ss["iter_k"] == p["iter_k"]


@pytest.mark.parametrize("step", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("kind", list(S.KINDS))
def test_every_step_runs_for_every_kind(step, kind):
    at = _run(step=step, kind_select=kind, m_slider=6, n_slider=6)
    _ok(at)
    assert at.session_state["pdl_step"] == step


@pytest.mark.parametrize("off", [None] + list(C.BLOCK_KEYS))
def test_every_block_switch_runs_on_every_step(off):
    for step in (1, 2, 3):
        state = {} if off is None else {C.BLOCK_KEYS[off]: False}
        if off is None:
            state = {key: False for key in C.BLOCK_KEYS.values()}                              # alle aus: Grundform
        _ok(_run(step=step, kind_select="textbook", **state))
        _ok(_run(step=step, m_slider=8, n_slider=8, cap_select=0, **state))


def test_iteration_slider_on_two_variables_and_every_position():
    at = _run(step=1, kind_select="textbook")
    _ok(at)
    total = at.session_state["iter_k"]
    assert total > 10 and len(at.slider) >= 1
    for k in (0, 1, 9, total // 2, total):
        at.session_state["iter_k"] = k
        at.run()
        _ok(at)
    assert any("Iteration" in md.value for md in at.markdown)
    three = _run(step=1, kind_select="centre")
    _ok(three)
    assert "iter_k" not in three.session_state and any("kein Bild des Wegs" in md.value for md in three.markdown)


def test_status_messages_reflect_the_run():
    hang = _run(step=2, m_slider=10, n_slider=10, scale_select=3, precond_toggle=False, cap_select=1)
    _ok(hang)
    assert any("Iterationsgrenze" in w.value and "kein Optimalitätsanspruch" in w.value for w in hang.warning)
    inf = _run(step=2, kind_select="infeasible")
    _ok(inf)
    assert any("unzulässig" in i.value and "Beweis" in i.value for i in inf.info)
    unb = _run(step=2, kind_select="unbounded")
    _ok(unb)
    assert any("unbeschränkt" in i.value and "Beweis" in i.value for i in unb.info)


def test_on_demand_experiments():
    at = _run(step=3, cap_select=0)
    _click(at, "ablation_start")
    _ok(at)
    assert any("Konfiguration" in d.value.columns and len(d.value) == 12 for d in at.dataframe) and len(at.get("plotly_chart")) == 1
    four = _run(step=4, eps_select=1)
    for key in ("size_start", "large_start", "cube_start"):
        _click(four, key)
        _ok(four)
    assert len(four.get("plotly_chart")) == 3 and any("PDLP / Simplex" in d.value.columns for d in four.dataframe)
    five = _run(step=5, kind_select="plateau", m_slider=8, n_slider=8, cap_select=0)
    for key in ("eps_start", "scale_start", "detect_start"):
        _click(five, key)
        _ok(five)
    assert len(five.get("plotly_chart")) >= 3 and any("Strahl gefunden (Beweis)" in d.value.columns for d in five.dataframe)


@pytest.mark.parametrize("kw", [dict(kind_select="random", m_slider=C.M_MIN, n_slider=C.N_MIN), dict(kind_select="random", m_slider=C.M_MAX, n_slider=C.N_MAX, density_select=len(C.DENSITIES) - 1),
                                dict(kind_select="mixed", m_slider=C.M_MIN, n_slider=C.N_MAX, density_select=0), dict(kind_select="plateau", m_slider=C.M_MAX, n_slider=C.N_MIN),
                                dict(kind_select="klee_minty", n_slider=S.CUBE_MAX)])
def test_extreme_sizes_run_on_every_step(kw):
    for step in (1, 2, 3, 4, 5):
        _ok(_run(step=step, cap_select=0, **kw))


@pytest.mark.parametrize("kw", [dict(eps_select=0), dict(eps_select=len(C.EPS_EXPS) - 1), dict(cap_select=0), dict(cap_select=len(C.CAPS) - 1), dict(scale_select=len(C.SCALE_EXPS) - 1), dict(density_select=0)])
def test_extreme_controls_on_every_kind(kw):
    for kind in ("textbook", "random", "plateau", "infeasible", "unbounded"):
        _ok(_run(step=2, kind_select=kind, m_slider=6, n_slider=6, **kw))


def test_dice_button_changes_the_seed():
    at = _run(kind_select="random")
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Instanz generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old and at.session_state["seed_widget"] == at.session_state["seed_input"]


def test_permalink_values_are_clamped_and_invalid_choices_fall_back_to_the_default():
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in dict(m="999", n="1", step="9", kind="nope", dens="99", eps="-1", cap="99", scale="-4", avg="ja", restart="2", pre="x", adapt="", pw="true", seed="-4").items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    ss = at.session_state
    assert (ss["m_slider"], ss["n_slider"], ss["pdl_step"], ss["kind_select"], ss["density_select"], ss["eps_select"], ss["cap_select"], ss["scale_select"], ss["average_toggle"], ss["restart_toggle"], ss["precond_toggle"],
            ss["adaptive_toggle"], ss["pw_toggle"], ss["seed_input"]) == (C.M_MAX, C.N_MIN, 2, "random", len(C.DENSITIES) - 1, 0, len(C.CAPS) - 1, 0, True, True, True, True, True, 0)


def test_permalink_accepts_valid_values():
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in dict(kind="plateau", m="10", n="9", seed="7", dens="1", eps="3", cap="1", scale="2", avg="0", restart="1", pre="0", adapt="1", pw="0", step="4").items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    ss = at.session_state
    assert (ss["kind_select"], ss["m_slider"], ss["n_slider"], ss["seed_input"], ss["density_select"], ss["eps_select"], ss["cap_select"], ss["scale_select"], ss["average_toggle"], ss["restart_toggle"], ss["precond_toggle"],
            ss["adaptive_toggle"], ss["pw_toggle"], ss["pdl_step"]) == ("plateau", 10, 9, 7, 1, 3, 1, 2, False, True, False, True, False, 4)
    assert at.query_params["avg"] in (["0"], "0") and at.query_params["restart"] in (["1"], "1")


def test_sidebar_shows_the_controls_that_belong_to_the_instance():
    fixed = _run(kind_select="centre")
    assert not any(w.key in ("m_widget", "n_widget") for w in fixed.slider) and not any(s.key == "density_widget" for s in fixed.select_slider)
    rnd = _run(kind_select="random")
    assert any(w.key == "m_widget" for w in rnd.slider) and any(w.key == "n_widget" for w in rnd.slider) and any(s.key == "density_widget" for s in rnd.select_slider)
    cube = _run(kind_select="klee_minty")
    assert any(w.key == "n_widget" for w in cube.slider) and not any(w.key == "m_widget" for w in cube.slider) and not any(s.key == "density_widget" for s in cube.select_slider)
    assert all(any(s.key == key for s in rnd.select_slider) for key in ("eps_select", "cap_select", "scale_select"))


def test_changing_kind_and_step_on_later_steps_does_not_crash():
    for step in (1, 2, 3, 4, 5):
        at = _run(step=step)
        _ok(at)
        for kw in (dict(kind_select="mixed", m_slider=8, n_slider=8), dict(kind_select="infeasible"), dict(kind_select="unbounded"), dict(kind_select="textbook"), dict(kind_select="plateau", m_slider=6, n_slider=8),
                   dict(kind_select="klee_minty", n_slider=8), dict(kind_select="random", m_slider=4, n_slider=3, average_toggle=False)):
            for k, v in kw.items():
                at.session_state[k] = v
            at.run()
            _ok(at)


def test_footer_limits_and_literature_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    assert any("Applegate" in m.value and "Chambolle" in m.value and "cuPDLP" in m.value for e in at.expander for m in e.markdown)
    assert set(P.BLOCKS) == set(C.BLOCK_KEYS) == set(C.BLOCK_HELP)
