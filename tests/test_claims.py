"""Jede Zahl aus README und App über die echten Auswertungsfunktionen (Iterationszahlen und Numerik-Grenzfälle nur als Bänder: Windows und Linux können abweichen)."""

import numpy as np

import pdl_algorithm as A
import pdl_constants as C
import pdl_evaluation as ev
import pdl_pdlp as P
import pdl_scenario as S
from pdl_evaluation import Settings

TWENTY = Settings(m=20, n=20)


def test_readme_ablation_on_twenty_by_twenty_random_instances():
    rows = {r["label"]: r for r in ev.ablation(TWENTY)}
    assert rows["Grundform"]["iterations"] == 20000 and rows["Grundform"]["optimal"] == 0
    assert 200 <= rows["PDLP (alle fünf)"]["iterations"] <= 400 and rows["PDLP (alle fünf)"]["optimal"] == 5
    assert 2000 <= rows["+ Vorkonditionierung"]["iterations"] <= 5000 and rows["+ Vorkonditionierung"]["optimal"] == 5
    assert 8000 <= rows["+ adaptive Schrittweite"]["iterations"] <= 20000 and rows["+ adaptive Schrittweite"]["optimal"] >= 3
    assert rows["+ Mittelung"]["optimal"] == 0 and rows["+ Neustarts"]["optimal"] == 0 and rows["+ Primalgewicht"]["optimal"] <= 2
    assert rows["+ Neustarts"]["iterations"] == rows["Grundform"]["iterations"] == rows["+ Mittelung"]["iterations"]
    assert 600 <= rows["ohne Mittelung"]["iterations"] <= 1500 and 700 <= rows["ohne Neustarts"]["iterations"] <= 1700 and 1100 <= rows["ohne Vorkonditionierung"]["iterations"] <= 2600
    assert 500 <= rows["ohne adaptive Schrittweite"]["iterations"] <= 1200 and 350 <= rows["ohne Primalgewicht"]["iterations"] <= 900
    assert 1.5 <= rows["ohne adaptive Schrittweite"]["matvecs"] / rows["PDLP (alle fünf)"]["matvecs"] <= 3.5 and rows["PDLP (alle fünf)"]["matvecs"] / rows["PDLP (alle fünf)"]["iterations"] > rows["ohne adaptive Schrittweite"]["matvecs"] / rows["ohne adaptive Schrittweite"]["iterations"]


def test_readme_restart_alone_is_the_basic_iteration_exactly_and_needs_the_average():
    inst = S.generate("random", 12, 12, 0.5, 35)
    base = P.pdlp(inst, eps=1e-4, max_iter=20000, average=False, restart=False, precond=False, adaptive=False, primal_weight=False)
    only = P.pdlp(inst, eps=1e-4, max_iter=20000, average=False, restart=True, precond=False, adaptive=False, primal_weight=False)
    assert only.iterations == base.iterations and only.x == base.x and only.restarts >= 1
    both = P.pdlp(inst, eps=1e-4, max_iter=20000, average=True, restart=True, precond=False, adaptive=False, primal_weight=False)
    assert both.iterations < base.iterations / 2


def test_readme_the_rejection_shrink_halves_the_matrix_vector_products(monkeypatch):
    insts = [S.centre_instance(), S.generate("mixed", 8, 8, 0.5, 1), S.generate("random", 20, 20, 0.3, 2), S.generate("random", 12, 12, 0.6, 3), S.klee_minty_instance(8), S.generate("mixed", 16, 20, 0.3, 4)]
    totals = {}
    for shrink in (1.0, 0.9):
        monkeypatch.setattr(P, "REJECT_SHRINK", shrink)
        rs = [P.pdlp(i, eps=1e-6) for i in insts]
        assert all(r.status == "optimal" for r in rs)
        totals[shrink] = (sum(r.matvecs for r in rs), sum(r.rejected for r in rs), sum(r.iterations for r in rs))
    assert totals[1.0][0] > 1.7 * totals[0.9][0] and totals[1.0][1] > 5 * totals[0.9][1] and totals[0.9][2] > 0.9 * totals[1.0][2]


def test_readme_accuracy_ladder_and_the_support_of_the_solution():
    rows = {r["k"]: r for r in ev.eps_sweep(TWENTY)}
    assert 40 <= rows[2]["iterations"] <= 120 and 200 <= rows[4]["iterations"] <= 400 and 500 <= rows[6]["iterations"] <= 850 and 750 <= rows[8]["iterations"] <= 1200 and 900 <= rows[10]["iterations"] <= 1500
    assert rows[10]["iterations"] < 30 * rows[2]["iterations"] and all(rows[k]["optimal"] == 5 for k in rows)
    assert rows[2]["nnz_x"] > 20 and rows[4]["nnz_x"] > 20 and rows[8]["nnz_x"] <= 21 and rows[10]["nnz_x"] == 20 and all(r["nnz_vertex"] == 20 and r["m"] == 20 for r in rows.values())
    plateau = ev.eps_sweep(Settings("plateau", 10, 12))
    assert all(r["nnz_vertex"] == 10 and r["nnz_x"] >= 20 for r in plateau)


def test_readme_scaling_over_the_column_spread():
    sw = ev.scale_sweep(Settings())
    rows = {r["k"]: r for r in sw["rows"]}
    basic, no_pre, full = sw["labels"]
    assert [rows[k][full]["right"] for k in (0, 2, 4, 6, 8)] == [5] * 5 and 3 <= rows[10][full]["right"] <= 5 and 2 <= rows[12][full]["right"] <= 4
    assert rows[0][basic]["right"] <= 2 and rows[0][no_pre]["right"] == 5 and all(rows[k][basic]["right"] == 0 and rows[k][no_pre]["right"] == 0 for k in (2, 4, 6, 8, 10, 12))
    assert 200 <= rows[0][full]["iterations"] <= 500 and rows[12][full]["iterations"] > 10 * rows[0][full]["iterations"] and rows[4][full]["iterations"] < rows[8][full]["iterations"]


def test_readme_size_sweep_and_large_sparse_instances():
    sw = ev.size_sweep(Settings())
    rows = {r["n"]: r for r in sw["rows"]}
    assert 6 <= rows[8]["ratio_ipm"] <= 10 and rows[64]["ratio_ipm"] < 1.0 < rows[32]["ratio_ipm"] and sw["cross_ipm"] == 64 and sw["cross_simplex"] is None
    assert 80 <= rows[8]["ratio_simplex"] <= 160 and 5 <= rows[100]["ratio_simplex"] <= 12
    assert 150 <= rows[8]["it_pdlp"] <= 320 and 1000 <= rows[100]["it_pdlp"] <= 1600 and all(rows[n]["it_ipm"] <= 12 for n in rows) and rows[100]["pivots"] < 140
    sparse = ev.size_sweep(Settings(density_i=0))
    assert sparse["cross_ipm"] in (16, 32) and sparse["cross_simplex"] is None
    big = {r["n"]: r for r in ev.large_sparse(Settings())}
    assert all(4 <= big[n]["ratio"] <= 40 for n in (200, 300, 500)) and all(big[n]["pivots"] < 60 for n in big) and all(r["status"] == "optimal" for r in big.values())    # Iterationszahl plattformabhängig (n = 500: 12.0 unter Windows, 18.1 unter Linux mit neuerem numpy)


def test_readme_klee_minty_cube_and_detection():
    rows = {r["n"]: r for r in ev.cube_sweep(Settings())}
    assert all(30 <= r["it_pdlp"] <= 130 for r in rows.values()) and rows[14]["it_pdlp"] < rows[2]["it_pdlp"] + 20 and rows[2]["it_ipm"] <= 6 and 12 <= rows[14]["it_ipm"] <= 18
    assert rows[14]["pivots"] == 16383 and rows[14]["flops_simplex"] == 14253210 and rows[14]["flops_pdlp"] < rows[14]["flops_ipm"] < rows[14]["flops_simplex"]
    det = ev.detection(Settings())
    assert det["infeasible"]["found"] == 30 and det["unbounded"]["found"] == 30 and 150 <= det["infeasible"]["iterations"] <= 700 and 30 <= det["unbounded"]["iterations"] <= 300


def test_readme_the_optimum_matches_the_simplex_on_a_hundred_fresh_instances():
    ok = 0
    for sd in range(100):
        inst = S.generate("mixed", 8, 8, 0.5, 1000 + sd)
        ref = A.solve(inst)
        r = P.pdlp(inst, eps=1e-6)
        ok += ref.status == "optimal" and r.status == "optimal" and ev.rel_error(r.obj, ref.obj) < 1e-4
    assert ok >= 98
    assert C.BLOCK_KEYS.keys() == set(P.BLOCKS) and np.isclose(P.NORM_ITERS, 40)


def test_readme_the_basic_form_does_not_reach_one_millionth_within_the_cap():
    for m in (8, 20):
        for _sd, _base, inst in ev.family(Settings(m=m, n=m)):
            r = P.pdlp(inst, eps=1e-6, max_iter=20000, average=False, restart=False, precond=False, adaptive=False, primal_weight=False)
            assert r.status == "limit" and r.iterations == 20000
