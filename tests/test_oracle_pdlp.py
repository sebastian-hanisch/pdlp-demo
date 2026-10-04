"""Unabhängige Orakel für PDLP: (1) HiGHS (Optimalwert), (2) die Residuen des Ergebnisses am Original neu gerechnet (eigene Formel aus x und y), (3) PDHG-Iterierte einer eigenen Zehn-Zeilen-Schleife
gegen die Grundform der Demo, (4) Ruiz- und Pock-Chambolle-Skalierung in eigener Schleife, (5) Status konstruierter unzulässiger und unbeschränkter Instanzen gegen HiGHS."""

import random

import numpy as np
import pytest

import pdl_pdlp as P
import pdl_scenario as S

linprog = pytest.importorskip("scipy.optimize").linprog
LE, GE, EQ = S.LE, S.GE, S.EQ


def _lp(inst):
    Am, b, c = inst.arrays()
    ub = [i for i, s in enumerate(inst.senses) if s != EQ]
    eq = [i for i, s in enumerate(inst.senses) if s == EQ]
    sg = np.array([1.0 if inst.senses[i] == LE else -1.0 for i in ub])
    return linprog(-c, A_ub=(Am[ub] * sg[:, None]) if ub else None, b_ub=(b[ub] * sg) if ub else None, A_eq=Am[eq] if eq else None, b_eq=b[eq] if eq else None, bounds=[(0, None)] * inst.n, method="highs")


def _instances():
    yield S.textbook_instance()
    yield S.centre_instance()
    yield S.degenerate_instance()
    for i in range(9):
        yield S.generate(("random", "mixed", "plateau")[i % 3], 3 + i % 6, 3 + (i * 2) % 6, 0.5, i)


def test_optimum_and_original_residuals_against_highs_and_an_independent_formula():
    solved = 0
    for inst in _instances():
        ref = _lp(inst)
        assert ref.status == 0
        for kw in ({}, {"precond": False}, {"average": False, "restart": False}):
            r = P.pdlp(inst, eps=1e-6, **kw)
            if r.status != "optimal":
                assert r.status == "limit"
                continue
            assert r.obj == pytest.approx(-ref.fun, rel=5e-5, abs=1e-4)
            M, b, c, _n, _ = P.pdlp_form(inst)
            x, y = np.array(r.x_full), np.array(r.y)
            pr = np.linalg.norm(M @ x - b) / (1 + np.linalg.norm(b))
            dr = np.linalg.norm(np.maximum(M.T @ y - c, 0.0)) / (1 + np.linalg.norm(c))
            gap = abs(c @ x - b @ y) / (1 + abs(c @ x) + abs(b @ y))
            assert max(pr, dr, gap) <= 1.0001e-6 and x.min() >= -1e-12
            solved += 1
    assert solved >= 30


def test_basic_pdhg_iterates_equal_an_independent_loop():
    for inst in list(_instances())[:8]:
        M, b, c, n, _ = P.pdlp_form(inst)
        eta = 0.9 / P.MatOp(M).norm()                                                        # Schrittweite 0.9 / ||M|| (Norm aus der Potenzmethode der Demo, gegen SVD unten geprüft)
        assert eta == pytest.approx(0.9 / np.linalg.norm(M, 2), rel=0.02)
        r = P.pdlp(inst, eps=1e-30, max_iter=120, average=False, restart=False, precond=False, adaptive=False, primal_weight=False, keep=True)
        x, y = np.zeros(M.shape[1]), np.zeros(M.shape[0])
        for _ in range(120):
            xn = np.maximum(0.0, x - eta * (c - M.T @ y))
            y = y + eta * (b - M @ (2 * xn - x))
            x = xn
        assert np.allclose(x[:n], np.array(r.points[-1]), atol=1e-9, rtol=1e-9)


def test_preconditioning_equals_an_independent_ruiz_and_pock_chambolle_loop():
    for seed in range(8):
        Mx = S.generate("mixed", 3 + seed % 5, 3 + seed % 4, 0.5, seed).arrays()[0]
        A = np.hstack([Mx, np.eye(Mx.shape[0])])
        Mp, rr, ss = P.precondition(A)
        B, r, s = A.copy(), np.ones(A.shape[0]), np.ones(A.shape[1])
        for norm in [lambda Z, ax: np.abs(Z).max(axis=ax)] * 10 + [lambda Z, ax: np.abs(Z).sum(axis=ax)]:
            a, d = np.sqrt(norm(B, 1)), np.sqrt(norm(B, 0))
            a[a == 0], d[d == 0] = 1.0, 1.0
            B, r, s = B / a[:, None] / d, r * a, s * d
        assert np.allclose(B, Mp) and np.allclose(r, rr) and np.allclose(s, ss) and np.allclose(A, Mp * rr[:, None] * ss)


def test_constructed_infeasible_and_unbounded_instances_get_the_status_of_highs():
    for seed in range(12):
        rng = random.Random(seed)
        n = rng.randint(2, 5)
        rows = [[rng.uniform(0.5, 3) for _ in range(n)] for _ in range(3)]
        b = [rng.uniform(5, 20) for _ in range(3)]
        rows.append(list(rows[0]))
        b.append(b[0] + rng.uniform(1, 5))
        inf = S.Instance(tuple(map(tuple, rows)), tuple(b), (1.0,) * n, (LE, LE, LE, GE), tuple(f"x{j}" for j in range(n)), tuple(f"r{i}" for i in range(4)), "x")
        assert _lp(inf).status == 2 and P.pdlp(inf, max_iter=3000).status == "infeasible"
        rows = [[rng.uniform(0.5, 3) for _ in range(n - 1)] + [0.0] for _ in range(2)]
        unb = S.Instance(tuple(map(tuple, rows)), tuple(rng.uniform(5, 20) for _ in range(2)), (1.0,) * n, (LE, LE), tuple(f"x{j}" for j in range(n)), ("r0", "r1"), "x")
        assert _lp(unb).status == 3 and P.pdlp(unb, max_iter=3000).status == "unbounded"
