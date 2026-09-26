"""Auswertung: Einstellungen, Analyse gegen Referenz, Ablation, Genauigkeitsleiter, Größe, große dünne Instanzen, Würfel, Erkennung (Skalierungs-Sweep: siehe test_claims)."""

import pytest

import pdl_constants as C
import pdl_evaluation as ev
import pdl_pdlp as P
import pdl_scenario as S
from tests.test_scenario import _highs, reference_status


def test_settings_clamp_the_index_controls_and_expose_the_blocks():
    s = ev.Settings(density_i=99, eps_i=-3, cap_i=99, scale_i=99)
    assert s.density == 1.0 and s.eps == 1e-2 and s.cap == 100000 and s.scale_exp == 12
    d = ev.Settings()
    assert d.density == 0.5 and d.eps == 1e-4 and d.cap == 20000 and d.scale_exp == 0 and d.blocks == ev.FULL and ev.Settings(average=False).blocks["average"] is False
    assert set(ev.FULL) == set(ev.BASIC) == set(P.BLOCKS)


def test_std_nnz_counts_structure_and_slack_components():
    inst = S.textbook_instance()
    assert ev.std_nnz(inst, (2.0, 6.0)) == 3                                            # x1, x2 und der Schlupf der ersten Zeile (4 - 2 = 2); die beiden anderen Zeilen sind bindend
    assert ev.std_nnz(inst, (0.0, 0.0)) == 3                                            # nur die drei Schlupfvariablen
    assert ev.rel_error(10.0, 10.0) == 0.0 and ev.rel_error(11.0, 10.0) == pytest.approx(1 / 11)


def test_analyse_compares_the_chosen_blocks_with_basic_full_mehrotra_and_the_simplex_and_is_cached():
    a = ev.analyse(ev.Settings())
    assert a is ev.analyse(ev.Settings())
    assert a.res is a.full and a.res.status == "optimal" and a.ref_status == "optimal" and a.error < 1e-3 and a.basic.iterations > 10 * a.res.iterations
    assert a.ipm.status == "optimal" and a.simplex.status == "optimal" and a.flops_simplex > 0 and a.flops_pdlp == a.res.flops
    off = ev.analyse(ev.Settings(average=False, restart=False, precond=False, adaptive=False, primal_weight=False, kind="textbook", eps_i=2))
    assert off.res is off.basic and off.res.status == "optimal" and len(off.res.points) == off.res.iterations + 1
    bad = ev.analyse(ev.Settings(scale_i=3, precond=False, m=10, n=10))
    assert bad.res.status == "limit" and bad.error > 0.05 and bad.full.status == "optimal"
    for kind, want in (("infeasible", "infeasible"), ("unbounded", "unbounded")):
        assert ev.analyse(ev.Settings(kind)).res.status == want and ev.analyse(ev.Settings(kind)).ref_status == want


def test_family_uses_five_fixed_seeds_for_random_kinds_and_one_instance_otherwise():
    assert [sd for sd, _b, _i in ev.family(ev.Settings())] == list(C.SWEEP_SEEDS)
    assert len(ev.family(ev.Settings("centre"))) == 1 and len(ev.family(ev.Settings("plateau"))) == 5
    assert ev.family(ev.Settings(scale_i=2))[0][2] != ev.family(ev.Settings())[0][2]


def test_ablation_has_twelve_configurations_and_restart_alone_changes_nothing():
    s = ev.Settings(m=8, n=8)
    rows = ev.ablation(s)
    assert len(rows) == 12 and [r["kind"] for r in rows].count("alone") == 5 and [r["kind"] for r in rows].count("removed") == 5 and rows[0]["label"] == "Grundform" and rows[6]["label"] == "PDLP (alle fünf)"
    by = {r["label"]: r for r in rows}
    assert by["PDLP (alle fünf)"]["optimal"] == 5 and by["PDLP (alle fünf)"]["iterations"] < by["Grundform"]["iterations"] / 10
    assert by["+ Neustarts"]["iterations"] == by["Grundform"]["iterations"] and by["+ Neustarts"]["matvecs"] == by["Grundform"]["matvecs"]
    assert by["+ Vorkonditionierung"]["iterations"] < by["Grundform"]["iterations"] / 3
    assert all(by[f"ohne {P.BLOCK_LABELS[b]}"]["iterations"] > by["PDLP (alle fünf)"]["iterations"] for b in P.BLOCKS)
    assert all(r["cap"] == 20000 and r["runs"] == 5 for r in rows)


def test_eps_sweep_iterations_grow_slowly_with_the_digits_and_report_the_support():
    rows = ev.eps_sweep(ev.Settings(m=8, n=8))
    assert [r["k"] for r in rows] == list(C.EPS_SWEEP_EXPS) and all(r["optimal"] == 5 for r in rows)
    its = [r["iterations"] for r in rows]
    assert its == sorted(its) and its[-1] < 30 * its[0] and all(r["m"] == 8 and r["nnz_vertex"] <= 8 for r in rows) and rows[-1]["nnz_x"] <= 9
    fixed = ev.eps_sweep(ev.Settings("centre"))
    assert all(r["runs"] == 1 for r in fixed)


def test_size_sweep_pdlp_beats_dense_interior_points_only_for_larger_sizes_and_never_the_simplex():
    sw = ev.size_sweep(ev.Settings())
    rows = sw["rows"]
    assert [r["n"] for r in rows] == list(C.SIZE_SIZES) and all(r["optimal"] == len(C.SIZE_SEEDS) for r in rows)
    assert rows[0]["ratio_ipm"] > 1.0 > rows[-1]["ratio_ipm"] and sw["cross_ipm"] in (32, 64, 100) and sw["cross_simplex"] is None
    assert all(r["ratio_simplex"] > 1.0 for r in rows) and rows[0]["ratio_simplex"] > rows[-1]["ratio_simplex"]
    assert all(r["it_ipm"] < 15 for r in rows)


def test_large_sparse_and_cube_and_detection():
    rows = ev.large_sparse(ev.Settings())
    assert [r["n"] for r in rows] == list(C.LARGE_SIZES) and all(r["status"] == "optimal" and r["ratio"] > 1.0 and r["pivots"] < 60 for r in rows)
    cube = ev.cube_sweep(ev.Settings())
    assert [r["n"] for r in cube] == list(C.CUBE_SIZES) and all(r["status"] == "optimal" and r["pivots"] == 2 ** r["n"] - 1 for r in cube)
    assert max(r["it_pdlp"] for r in cube) < 200 and cube[-1]["it_pdlp"] < 100 and cube[-1]["flops_pdlp"] < cube[-1]["flops_simplex"] / 10
    det = ev.detection(ev.Settings())
    assert det["infeasible"]["found"] + det["infeasible"]["limit"] + det["infeasible"]["other"] == 30 and det["infeasible"]["found"] >= 25 and det["unbounded"]["found"] >= 25


def test_constructed_instances_have_the_status_they_claim():
    for sd in range(6):
        assert reference_status(ev.constructed_infeasible(sd)) == "infeasible"
        assert reference_status(ev.constructed_unbounded(sd)) == "unbounded"
    assert _highs(S.generate("plateau", 10, 12, 0.5, 35)).status == 0
