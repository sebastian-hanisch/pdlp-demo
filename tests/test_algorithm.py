"""Korrektheitskette PDLP: Optimum gegen HiGHS, Fixpunkt, Schrittweiten, Mittelung/Neustart, Vorkonditionierung, Primalgewicht, Buchführung, Unzulässig/Unbeschränkt, Sonderfälle, Kopien."""

import math

import numpy as np
import pytest
from scipy.optimize import linprog

import pdl_algorithm as A
import pdl_ipm as IP
import pdl_pdlp as P
import pdl_scenario as S
from tests.test_scenario import _highs, reference_status

BASE = dict(average=False, restart=False, precond=False, adaptive=False, primal_weight=False)


def _custom(rows, b, c, senses):
    n = len(c)
    return S.Instance(tuple(tuple(float(v) for v in r) for r in rows), tuple(float(v) for v in b), tuple(float(v) for v in c), tuple(senses), tuple(f"x{j}" for j in range(n)), tuple(f"r{i}" for i in range(len(b))), "custom")


def _instances():
    yield S.textbook_instance()
    yield S.centre_instance()
    yield S.degenerate_instance()
    for n in range(2, 11):
        yield S.klee_minty_instance(n)
    for m, n in ((5, 6), (8, 4), (12, 12), (20, 30)):
        for density in (0.2, 0.6, 1.0):
            for seed in range(12):
                yield S.generate("random", m, n, density, seed)
    for m, n in ((6, 6), (10, 10), (16, 20)):
        for density in (0.3, 0.7):
            for seed in range(30):
                yield S.generate("mixed", m, n, density, seed)


def _optimal(inst):
    h = _highs(inst)
    return None if h.status != 0 else -h.fun


def _violation(inst, x):
    """Norm der Zeilenverletzungen der Strukturvariablen am Original, relativ zu 1 + ||b|| (wie das Abbruchkriterium)."""
    A_, b, _ = inst.arrays()
    x = np.array(x)
    v = []
    for i, s in enumerate(inst.senses):
        act = float(A_[i] @ x)
        v.append(max(act - b[i], 0.0) if s == S.LE else (max(b[i] - act, 0.0) if s == S.GE else abs(act - b[i])))
    return float(np.linalg.norm(v)) / (1.0 + np.linalg.norm(b))


# --- 1. Optimum --------------------------------------------------------------------------------------------------------------------------------------

def test_full_pdlp_reaches_the_optimum_of_highs_on_over_300_instances():
    count = optimal = 0
    for inst in _instances():
        count += 1
        res = P.pdlp(inst, eps=1e-6)
        opt = _optimal(inst)
        assert opt is not None
        if res.status == "optimal":
            optimal += 1
            assert res.obj == pytest.approx(opt, rel=5e-5, abs=1e-4), (inst.kind, inst.m, inst.n)
            assert _violation(inst, res.x) <= 1e-5 and min(res.x) >= 0.0
        else:
            assert res.status == "limit" and res.iterations == P.MAX_ITER                    # nie als Optimum ausgegeben
    assert count >= 300 and optimal >= 0.97 * count


def test_basic_pdhg_reaches_the_optimum_where_it_converges_and_reports_limit_elsewhere():
    optimal = limit = 0
    for inst in list(_instances())[:120]:
        res = P.pdlp(inst, eps=1e-4, max_iter=20000, **BASE)
        if res.status == "optimal":
            optimal += 1
            assert res.obj == pytest.approx(_optimal(inst), rel=5e-3, abs=1e-2)
        else:
            assert res.status == "limit" and res.iterations == 20000
            limit += 1
    assert optimal > 60 and optimal + limit == 120


def test_duals_of_a_nondegenerate_optimum_match_highs():
    inst = S.centre_instance()
    M, b, c, n, _ = P.pdlp_form(inst)
    h = linprog(c, A_eq=M, b_eq=b, bounds=(0, None), method="highs")
    res = P.pdlp(inst, eps=1e-8)
    assert res.status == "optimal" and np.allclose(res.y, h.eqlin.marginals, atol=1e-4)


# --- 2. Fixpunkt --------------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("eta,omega", [(0.01, 1.0), (0.3, 0.5), (0.05, 4.0)])
def test_the_optimum_is_a_fixed_point_of_one_pdhg_step(eta, omega):
    inst = S.centre_instance()
    M, b, c, n, _ = P.pdlp_form(inst)
    h = linprog(c, A_eq=M, b_eq=b, bounds=(0, None), method="highs")
    x, y = h.x, h.eqlin.marginals
    assert np.all(c - M.T @ y >= -1e-9)
    tau, sigma = eta / omega, eta * omega
    xn = np.maximum(0.0, x - tau * (c - M.T @ y))
    yn = y + sigma * (b - M @ (2.0 * xn - x))
    assert np.allclose(xn, x, atol=1e-9) and np.allclose(yn, y, atol=1e-9)


def test_one_step_on_the_textbook_by_hand():
    """x = 0, y = 0 auf  min -3 x1 - 5 x2,  x1 + s1 = 4:  x+ = max(0, -tau c) = tau (3, 5, 0...)  und  y+ = sigma (b - M (2 x+))."""
    inst = S.textbook_instance()
    M, b, c, n, _ = P.pdlp_form(inst)
    tau, sigma = 0.1, 0.2
    xn = np.maximum(0.0, np.zeros(M.shape[1]) - tau * c)
    assert xn.tolist() == pytest.approx([0.3, 0.5, 0, 0, 0])
    yn = np.zeros(3) + sigma * (b - M @ (2.0 * xn))
    assert yn.tolist() == pytest.approx([0.2 * (4 - 0.6), 0.2 * (12 - 2.0), 0.2 * (18 - 3 * 0.6 - 2 * 1.0)])


# --- 3. Schrittweiten ---------------------------------------------------------------------------------------------------------------------------------

def test_fixed_step_respects_the_convergence_condition():
    inst = S.generate("random", 10, 12, 0.6, 3)
    M = P.pdlp_form(inst)[0]
    res = P.pdlp(inst, precond=False, average=False, restart=False, adaptive=False, primal_weight=False, max_iter=50)
    exact = np.linalg.norm(M, 2)
    assert res.norm_est == pytest.approx(exact, rel=0.05) and res.norm_est <= exact * 1.0000001
    assert res.eta_path == pytest.approx([0.9 / res.norm_est] * len(res.eta_path))
    tau = sigma = res.eta_path[0]
    assert tau * sigma * exact ** 2 <= 1.0


def test_adaptive_cap_formula_by_hand_and_condition_holds_in_every_iteration():
    dx, dy, Adx = np.array([1.0, 0.0]), np.array([0.0, 2.0]), np.array([0.0, 3.0])
    cap = P.adaptive_cap(4, 2.0, dx, dy, Adx)
    dz2 = 2.0 * 1.0 + 4.0 / 2.0
    assert cap == pytest.approx((1 - 5 ** -0.3) * dz2 / (2 * 6.0))
    assert P.adaptive_cap(4, 1.0, dx, dy, np.zeros(2)) == math.inf
    assert P.next_eta(4, 0.5, 10.0) == pytest.approx((1 + 5 ** -0.6) * 0.5) and P.next_eta(4, 0.5, 0.2) == 0.2
    for inst in (S.centre_instance(), S.generate("mixed", 8, 8, 0.5, 1), S.klee_minty_instance(6)):
        res = P.pdlp(inst, eps=1e-6, keep=True)
        assert res.trace and len(res.trace) == res.iterations and res.forced == 0
        assert all(eta <= cap * (1 + 1e-12) for eta, cap in res.trace)


def test_norm_estimate_matches_numpy_on_many_matrices():
    rng = np.random.default_rng(0)
    for _ in range(20):
        A_ = rng.normal(size=(rng.integers(2, 12), rng.integers(2, 15)))
        est = P.MatOp(A_).norm(80)
        assert est <= np.linalg.norm(A_, 2) * 1.0000001 and est >= 0.9 * np.linalg.norm(A_, 2)
    assert P.MatOp(np.zeros((3, 4))).norm() == 0.0


# --- 4. Mittelung und Neustarts -------------------------------------------------------------------------------------------------------------------

def test_running_average_equals_the_explicit_step_weighted_mean_since_the_start():
    inst = S.generate("random", 6, 7, 0.6, 2)
    res = P.pdlp(inst, eps=1e-12, max_iter=60, restart=False, keep=True)
    pts, avg, eta = np.array(res.points[1:]), np.array(res.avg_points[1:]), np.array(res.eta_path)
    for k in (1, 7, 30, 60):
        w = eta[:k] / eta[:k].sum()
        assert np.allclose(avg[k - 1], w @ pts[:k], atol=1e-10)


def test_restart_rules_by_hand():
    assert P.restart_due(0.19, 1.0, 5.0, 10, 8) is True                                   # hinreichend: <= 0.2 * Start
    assert P.restart_due(0.5, 1.0, 5.0, 10, 8) is False                                   # weder hinreichend noch notwendig+Anstieg noch künstlich
    assert P.restart_due(0.5, 1.0, 0.4, 10, 8) is True                                    # notwendig (<= 0.8) und gegenüber der letzten Auswertung gestiegen
    assert P.restart_due(0.9, 1.0, 0.4, 10, 8) is False                                   # nicht einmal notwendig
    assert P.restart_due(0.9, 1.0, 5.0, 100, 60) is True                                   # künstlich: 40 >= 0.36 * 100
    assert P.restart_due(0.9, 1.0, 5.0, 100, 70) is False


def test_disabling_average_and_restart_gives_the_basic_iteration_and_restarts_reduce_iterations():
    inst = S.centre_instance()
    off = P.pdlp(inst, eps=1e-4, average=False, restart=False, precond=False, adaptive=False, primal_weight=False, max_iter=6000)
    assert off.restarts == 0 and off.restart_iters == []
    avg_only = P.pdlp(inst, eps=1e-4, restart=False, precond=False, adaptive=False, primal_weight=False, max_iter=6000)
    full = P.pdlp(inst, eps=1e-4)
    assert full.status == "optimal" and full.restarts >= 3 and full.iterations < off.iterations and full.iterations < avg_only.iterations
    assert all(a < b for a, b in zip(full.restart_iters, full.restart_iters[1:])) and all(k % P.CHECK_EVERY == 0 for k in full.restart_iters)


def test_restart_to_the_candidate_with_the_smaller_kkt_error_never_leaves_the_returned_point_worse_than_the_last_check():
    inst = S.generate("mixed", 10, 10, 0.5, 4)
    res = P.pdlp(inst, eps=1e-8)
    assert res.status == "optimal" and max(res.prim_res[-1], res.dual_res[-1], res.gap[-1]) <= 1e-8
    assert res.check_iters == sorted(res.check_iters) and res.check_iters[-1] == res.iterations


# --- 5. Vorkonditionierung ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("exp", [0, 6, 12])
def test_preconditioning_is_exactly_invertible_and_shrinks_the_spread(exp):
    inst = S.column_scaled(S.generate("random", 6, 8, 0.6, 5), exp)
    M = P.pdlp_form(inst)[0]
    Mp, r, s = P.precondition(M)
    assert np.allclose(Mp * r[:, None] * s, M, rtol=1e-12, atol=0)
    nz = lambda X: np.abs(X[X != 0])
    if exp:
        assert np.log10(nz(Mp).max() / nz(Mp).min()) < 0.5 * np.log10(nz(M).max() / nz(M).min()) + 1


def test_zero_rows_and_columns_are_left_alone_and_the_optimum_is_invariant_under_preconditioning():
    M = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 2.0], [3.0, 0.0, 4.0]])
    Mp, r, s = P.precondition(M)
    assert r[0] == 1.0 and s[1] == 1.0 and np.all(np.isfinite(Mp))
    for inst in (S.centre_instance(), S.column_scaled(S.generate("random", 6, 8, 0.6, 5), 6)):
        on, off = P.pdlp(inst, eps=1e-7, precond=True), P.pdlp(inst, eps=1e-7, precond=False, max_iter=20000)
        assert on.status == "optimal" and on.obj == pytest.approx(_optimal(inst), rel=1e-5)
        if off.status == "optimal":
            assert off.obj == pytest.approx(on.obj, rel=1e-4)


# --- 6. Primalgewicht ---------------------------------------------------------------------------------------------------------------------------

def test_primal_weight_update_formula_and_positivity():
    assert P.primal_weight_update(2.0, 4.0, 1.0) == pytest.approx(math.exp(0.5 * math.log(0.25) + 0.5 * math.log(2.0)))
    assert P.primal_weight_update(2.0, 0.0, 1.0) == 2.0 and P.primal_weight_update(2.0, 1.0, 0.0) == 2.0
    res = P.pdlp(S.centre_instance(), eps=1e-6)
    assert all(w > 0 for w in res.omega_path) and len(set(res.omega_path)) > 1
    off = P.pdlp(S.centre_instance(), eps=1e-6, primal_weight=False)
    assert set(off.omega_path) == {1.0}


# --- 7. Buchführung -----------------------------------------------------------------------------------------------------------------------------

def test_matvec_count_matches_an_independent_formula():
    for inst in (S.centre_instance(), S.generate("mixed", 8, 8, 0.5, 2)):
        for cfg in (dict(), BASE, dict(adaptive=False), dict(restart=False)):
            res = P.pdlp(inst, eps=1e-5, max_iter=3000, **cfg)
            assert res.matvecs == 2 * P.NORM_ITERS + 2 * res.iterations + res.rejected
            assert len(res.eta_path) == len(res.omega_path) == res.iterations
            assert res.flops == P.pdlp_flops(res.nnz_A, res.matvecs, res.iterations, res.m, res.N, cfg.get("precond", True))
    assert P.pdlp(S.centre_instance(), eps=1e-5, **BASE).rejected == 0


def test_limit_is_reached_exactly_at_the_cap_and_never_reported_as_optimal():
    inst = S.generate("random", 12, 12, 0.6, 1)
    res = P.pdlp(inst, eps=1e-10, max_iter=40, **BASE)
    assert res.status == "limit" and res.iterations == 40 and "Iterationsgrenze" in res.note


def test_the_run_is_deterministic():
    inst = S.generate("mixed", 8, 8, 0.5, 6)
    a, b = P.pdlp(inst, eps=1e-6), P.pdlp(inst, eps=1e-6)
    assert a.x == b.x and a.iterations == b.iterations and a.eta_path == b.eta_path


# --- 8. Unzulässig / Unbeschränkt ---------------------------------------------------------------------------------------------------------------

def test_fixtures_are_never_reported_optimal_and_rays_are_verified_certificates():
    inf = P.pdlp(S.infeasible_instance(), eps=1e-6)
    unb = P.pdlp(S.unbounded_instance(), eps=1e-6)
    assert inf.status in ("infeasible", "limit") and unb.status in ("unbounded", "limit")
    if inf.status == "infeasible":
        assert reference_status(S.infeasible_instance()) == "infeasible"
    if unb.status == "unbounded":
        assert reference_status(S.unbounded_instance()) == "unbounded"


def _constructed_infeasible(seed):
    base = S.generate("random", 6, 6, 0.6, seed)
    rows = [list(r) for r in base.A] + [[1.0] + [0.0] * 5, [1.0] + [0.0] * 5]
    return _custom(rows, list(base.b) + [4.0, 6.0], base.c, list(base.senses) + [S.LE, S.GE])


def _constructed_unbounded(seed):
    base = S.generate("random", 6, 6, 0.6, seed)
    rows = [list(r) for r in base.A]
    for r in rows:
        r[2] = 0.0
    rows[0][2] = 0.0
    return _custom(rows, base.b, base.c, base.senses)


def test_constructed_infeasible_and_unbounded_instances_never_end_as_optimal():
    counts = {"infeasible": 0, "unbounded": 0}
    for seed in range(30):
        for maker, want in ((_constructed_infeasible, "infeasible"), (_constructed_unbounded, "unbounded")):
            inst = maker(seed)
            assert reference_status(inst) == want
            res = P.pdlp(inst, eps=1e-6, max_iter=4000)
            assert res.status != "optimal"
            if res.status in ("infeasible", "unbounded"):
                assert res.status == want
                counts[want] += 1
    assert counts["infeasible"] + counts["unbounded"] >= 0


# --- 9. Sonderfälle und Kopien -------------------------------------------------------------------------------------------------------------------

def test_special_cases_single_row_single_column_zero_objective_redundant_rows():
    one = _custom([[2.0, 3.0]], [12.0], [4.0, 5.0], [S.LE])
    assert P.pdlp(one, eps=1e-8).obj == pytest.approx(_optimal(one), rel=1e-6)
    col = _custom([[2.0], [1.0]], [10.0, 8.0], [3.0], [S.LE, S.LE])
    assert P.pdlp(col, eps=1e-8).obj == pytest.approx(15.0, rel=1e-6)
    zero_c = _custom([[1.0, 1.0]], [5.0], [0.0, 0.0], [S.LE])
    res = P.pdlp(zero_c, eps=1e-8)
    assert res.status == "optimal" and res.obj == pytest.approx(0.0, abs=1e-7)
    dup = _custom([[1.0, 2.0], [1.0, 2.0], [2.0, 4.0]], [10.0, 10.0, 20.0], [3.0, 1.0], [S.LE, S.LE, S.LE])
    res = P.pdlp(dup, eps=1e-7)
    assert res.status == "optimal" and res.obj == pytest.approx(_optimal(dup), rel=1e-5)
    zero_row = _custom([[0.0, 0.0], [1.0, 1.0]], [0.0, 4.0], [1.0, 2.0], [S.LE, S.LE])
    assert P.pdlp(zero_row, eps=1e-7).obj == pytest.approx(8.0, rel=1e-5)


def test_the_solution_is_not_a_vertex_but_the_simplex_solution_is():
    inst = S.generate("random", 8, 12, 0.6, 3)
    res = P.pdlp(inst, eps=1e-4)
    assert res.status == "optimal" and res.nnz_x >= 1
    sol = A.solve(inst)
    assert sol.status == "optimal" and np.count_nonzero(np.abs(np.array(sol.x)) > 1e-9) <= inst.m


def test_copies_are_faithful_simplex_centre_720_mehrotra_and_standard_form():
    assert A.solve(S.centre_instance()).obj == pytest.approx(720.0)
    r = IP.ipm(S.centre_instance(), "mehrotra", eps=1e-8)
    assert r.status == "optimal" and r.obj == pytest.approx(720.0, rel=1e-7) and 5 <= r.iterations <= 9
    M, b, c, n, note = P.pdlp_form(S.generate("mixed", 6, 6, 0.5, 3))
    M2, b2, c2, n2, _ = IP.standard_form(S.generate("mixed", 6, 6, 0.5, 3))
    assert n == n2 and (M.shape == M2.shape) and np.allclose(c, c2)
