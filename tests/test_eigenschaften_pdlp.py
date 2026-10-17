"""Eigenschaftstests (metamorph) für PDLP (ORACLE-PLAYBOOK §5.2). Aussagen, die für das Verfahren gelten müssen, ohne dass man das richtige Ergebnis kennt.

Das LP ist  max c·x,  A x (<=|>=|=) b,  x >= 0. PDLP ist ein Näherungsverfahren: Vergleiche des Zielwerts gelten mit der Toleranz des Abbruchs (eps = 1e-6, relativer Vergleich 1e-4);
ein Lauf, der nicht "optimal" meldet, wird nie als Optimum gewertet. Je Eigenschaft 12 bis 30 Zufallsinstanzen (Zufall, Mischung, Plateau; feste Seeds) plus Randfälle.

Eigenschaften:
 1. Ergebnis gleich HiGHS (scipy.optimize.linprog) innerhalb der Toleranz, primal zulässig bis zur Toleranz.
 2. Zielfunktion mit f > 0 multipliziert: Zielwert mit f skaliert. Umnummerieren der Variablen oder Bedingungen: derselbe Zielwert.
    Einheitenwechsel einer Variablen oder Zeile (Spalten- bzw. Zeilenskalierung): derselbe Zielwert.
 3. Redundante Zeile ändert das Optimum nicht, ein Schnitt verbessert es nie, eine zusätzliche Spalte verschlechtert es nie.
 4. Skaleninvarianz der Grundform (Applegate u. a. 2021, Anhang A, Proposition 1): c mit g_y, b mit g_x multipliziert, ohne Vorkonditionierung und Neustart und mit Primalgewicht ||c|| / ||b||:
    die Iterierten x skalieren exakt mit g_x (auch der Mittelwert), das Anfangsgewicht mit g_y / g_x.
 5. Die Residuen fallen im Mittel (Anfang gegen Ende, Abbruch bei <= eps); das Ergebnis erfüllt die Abbruchbedingung am Original.
 6. Neustarts: Zeitpunkte sind streng wachsende Vielfache der Prüfschrittweite, der erste liegt bei der ersten Prüfung, der Abstand zweier Neustarts ist durch das künstliche Kriterium (0.36 der
    bisherigen Iterationen) nach oben begrenzt; das Primalgewicht ändert sich nur unmittelbar nach einem Neustart (Applegate u. a., Abschnitt 3.3), ohne Primalgewicht bleibt es 1.
 7. Randfälle: keine Bedingungen, eine Variable, c = 0, doppelte Zeilen, Unzulässig und Unbeschränkt (nie "optimal", auch unter Skalierung und Umnummerieren).
 8. Gleiche Eingabe, gleiche Ausgabe (Iterationen, Lösung, Produkte)."""

import functools

import numpy as np
import pytest

import pdl_pdlp as P
import pdl_scenario as S

linprog = pytest.importorskip("scipy.optimize").linprog
LE, GE, EQ = S.LE, S.GE, S.EQ
EPS = 1e-6
REL = 1e-4


def _inst(rows, b, c, senses):
    n = len(c)
    return S.Instance(tuple(tuple(float(v) for v in r) for r in rows), tuple(float(v) for v in b), tuple(float(v) for v in c), tuple(senses),
                      tuple(f"x{j}" for j in range(n)), tuple(f"r{i}" for i in range(len(b))), "custom")


def _arrays(inst):
    A_, b, c = inst.arrays()
    return A_, b, c, list(inst.senses)


def _highs(inst):
    A_, b, c = inst.arrays()
    ub = [i for i, s in enumerate(inst.senses) if s != EQ]
    eq = [i for i, s in enumerate(inst.senses) if s == EQ]
    sg = np.array([1.0 if inst.senses[i] == LE else -1.0 for i in ub])
    return linprog(-c, A_ub=(A_[ub] * sg[:, None]) if ub else None, b_ub=(b[ub] * sg) if ub else None, A_eq=A_[eq] if eq else None, b_eq=b[eq] if eq else None,
                   bounds=[(0, None)] * inst.n, method="highs")


@functools.lru_cache(maxsize=None)
def _run(inst):
    return P.pdlp(inst, eps=EPS)


def _instances(count, m=6, n=8, offset=0):
    kinds = ("random", "mixed", "plateau")
    return [S.generate(kinds[i % 3], m, n, 0.6, offset + i) for i in range(count)]


def _same_value(r, ref_obj):
    return r.status == "optimal" and r.obj == pytest.approx(ref_obj, rel=REL, abs=REL)


def _violation(inst, x):
    A_, b, _c = inst.arrays()
    x = np.asarray(x, dtype=float)
    worst = max(0.0, float(-x.min())) if len(x) else 0.0
    for i in range(inst.m):
        act, s = float(A_[i] @ x), inst.senses[i]
        worst = max(worst, max(0.0, act - b[i]) if s == LE else (max(0.0, b[i] - act) if s == GE else abs(act - b[i])))
    return worst


# --- 1. gegen HiGHS ---------------------------------------------------------------------------------------------------------------------------------------


def test_result_equals_highs_within_tolerance_and_is_nearly_primal_feasible():
    for inst in _instances(16):
        ref = _highs(inst)
        assert ref.status == 0
        r = _run(inst)
        assert _same_value(r, -ref.fun), (inst.kind, r.status, r.obj, -ref.fun)
        b_scale = 1.0 + float(np.linalg.norm(inst.b))
        assert _violation(inst, r.x) <= 50 * EPS * b_scale                     # relative Residuen <= eps, Größenordnung der Verletzung


# --- 2. Skalierung, Umnummerieren, Einheitenwechsel -------------------------------------------------------------------------------------------------------


def test_objective_scaled_by_positive_factor_scales_the_value():
    rng = np.random.default_rng(11)
    for inst in _instances(12):
        base = _run(inst)
        A_, b, c, senses = _arrays(inst)
        f = float(rng.choice([0.1, 4.0, 60.0]))
        r = P.pdlp(_inst(A_, b, f * c, senses), eps=EPS)
        assert base.status == "optimal" and _same_value(r, f * base.obj)


def test_renumbering_variables_or_constraints_keeps_the_value():
    rng = np.random.default_rng(12)
    for inst in _instances(12):
        base = _run(inst)
        A_, b, c, senses = _arrays(inst)
        pv, pc = rng.permutation(A_.shape[1]), rng.permutation(A_.shape[0])
        rv = P.pdlp(_inst(A_[:, pv], b, c[pv], senses), eps=EPS)
        rc = P.pdlp(_inst(A_[pc], b[pc], c, [senses[i] for i in pc]), eps=EPS)
        assert base.status == "optimal" and _same_value(rv, base.obj) and _same_value(rc, base.obj)
        assert _violation(inst, np.array(rv.x)[np.argsort(pv)]) <= 50 * EPS * (1.0 + float(np.linalg.norm(b)))


def test_change_of_units_of_a_variable_or_a_row_keeps_the_value():
    rng = np.random.default_rng(13)
    for inst in _instances(6):
        base = _run(inst)
        A_, b, c, senses = _arrays(inst)
        lam, mu = rng.choice([0.05, 1.0, 20.0], size=A_.shape[1]), rng.choice([0.05, 1.0, 20.0], size=A_.shape[0])
        rc = P.pdlp(_inst(A_ * lam, b, c * lam, senses), eps=EPS)                  # x_j' = x_j / lam_j
        rr = P.pdlp(_inst(A_ * mu[:, None], b * mu, c, senses), eps=EPS)
        assert base.status == "optimal" and _same_value(rc, base.obj) and _same_value(rr, base.obj)


# --- 3. zusätzliche Zeilen und Spalten ------------------------------------------------------------------------------------------------------------------


def test_redundant_row_keeps_the_optimum_a_cut_never_improves_it_and_an_extra_column_never_hurts():
    rng = np.random.default_rng(14)
    for inst in _instances(7):
        base = _run(inst)
        A_, b, c, senses = _arrays(inst)
        assert base.status == "optimal"
        le = [i for i, s in enumerate(senses) if s == LE]
        lam = np.zeros(len(b))
        lam[le[0]], lam[le[-1]] = 0.7, 1.3                                           # nichtnegative Kombination von <=-Zeilen, rechte Seite etwas größer: folgt aus den Zeilen
        red = P.pdlp(_inst(np.vstack([A_, lam @ A_]), np.append(b, float(lam @ b) + 2.0), c, senses + [LE]), eps=EPS)
        assert _same_value(red, base.obj)
        dup = P.pdlp(_inst(np.vstack([A_, A_[1]]), np.append(b, b[1]), c, senses + [senses[1]]), eps=EPS)
        assert _same_value(dup, base.obj)
        a = rng.uniform(0.3, 3.0, size=A_.shape[1])
        beta = 0.7 * float(a @ np.array(base.x))
        cut = P.pdlp(_inst(np.vstack([A_, a]), np.append(b, beta), c, senses + [LE]), eps=EPS)
        assert cut.status in ("optimal", "limit", "infeasible")                      # bei Mischung kann der Schnitt unzulässig machen; nie "unbeschränkt"
        if cut.status == "optimal":
            assert cut.obj <= base.obj + REL * (1.0 + abs(base.obj))
        col = rng.uniform(0.0, 2.0, size=A_.shape[0]) * (rng.random(A_.shape[0]) < 0.7)
        more = P.pdlp(_inst(np.hstack([A_, col[:, None]]), b, np.append(c, float(rng.uniform(0.0, 5.0))), senses), eps=EPS)
        assert more.status == "optimal" and more.obj >= base.obj - REL * (1.0 + abs(base.obj))


# --- 4. Skaleninvarianz der Grundform (Anhang A der Arbeit) -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("average", [False, True])
def test_basic_form_is_scale_invariant_proposition_one(average):
    kw = dict(precond=False, adaptive=False, restart=False, average=average, primal_weight=True, keep=True, eps=1e-30, max_iter=40)
    for inst in _instances(8, offset=500):
        A_, b, c, senses = _arrays(inst)
        base = P.pdlp(inst, **kw)
        for gy, gx in ((3.0, 1.0), (1.0, 5.0), (0.2, 7.0)):
            r = P.pdlp(_inst(A_, b * gx, c * gy, senses), **kw)
            assert len(r.points) == len(base.points) == 41
            assert np.allclose(np.array(r.points), gx * np.array(base.points), rtol=1e-9, atol=1e-9 * gx)          # x_k' = g_x x_k
            if average:
                assert np.allclose(np.array(r.avg_points), gx * np.array(base.avg_points), rtol=1e-9, atol=1e-9 * gx)
            assert r.omega_path[0] == pytest.approx(base.omega_path[0] * gy / gx, rel=1e-12)                  # Anfangsgewicht ||c|| / ||b||


# --- 5. Residuen --------------------------------------------------------------------------------------------------------------------------------------------


def test_residuals_fall_on_average_and_the_result_meets_the_stopping_rule_at_the_original():
    long_runs = 0
    for inst in _instances(12, m=7, n=9, offset=600):
        r = _run(inst)
        assert r.status == "optimal"
        worst = np.maximum(np.maximum(r.prim_res, r.dual_res), r.gap)
        assert worst[-1] <= EPS < worst[0] or len(worst) == 1
        if len(worst) >= 9:
            long_runs += 1
            logs, half = np.log10(worst), len(worst) // 2
            assert logs[:half].mean() - logs[half:].mean() > 0.2 and np.polyfit(np.arange(len(logs)), logs, 1)[0] < 0   # im Mittel fallend (nicht monoton: Neustarts, Mittelwert)
        M, b, c, _n, _ = P.pdlp_form(inst)
        x, y = np.array(r.x_full), np.array(r.y)
        pr = np.linalg.norm(M @ x - b) / (1 + np.linalg.norm(b))
        dr = np.linalg.norm(np.maximum(M.T @ y - c, 0.0)) / (1 + np.linalg.norm(c))
        gap = abs(c @ x - b @ y) / (1 + abs(c @ x) + abs(b @ y))
        assert max(pr, dr, gap) <= EPS * 1.0001
    assert long_runs >= 8


# --- 6. Neustarts und Primalgewicht ------------------------------------------------------------------------------------------------------------------------


def test_restart_times_follow_the_artificial_rule_and_the_primal_weight_changes_only_after_restarts():
    for number, inst in enumerate(_instances(12, m=7, n=9, offset=600)):
        r = _run(inst)
        it = r.restart_iters
        assert r.restarts == len(it) and r.iterations > P.CHECK_EVERY
        assert all(k % P.CHECK_EVERY == 0 for k in it) and all(a < b for a, b in zip(it, it[1:]))
        assert it[0] == P.CHECK_EVERY                                              # seit Iteration 0 sind 100 % der Iterationen "lang": künstliches Kriterium schon bei der ersten Prüfung
        for a, b in zip(it, it[1:]):                                              # spätestens beim ersten Prüfpunkt k >= a / (1 - 0.36) greift das künstliche Kriterium
            assert b <= P.CHECK_EVERY * int(np.ceil(a / (1.0 - P.BETA_ARTIFICIAL) / P.CHECK_EVERY)) + P.CHECK_EVERY
        changes = {i for i in range(1, len(r.omega_path)) if r.omega_path[i] != r.omega_path[i - 1]}
        assert changes <= set(it)                                                  # Index i: Gewicht der Iteration i + 1, geändert direkt nach dem Neustart bei Iteration i
        assert all(w > 0 for w in r.omega_path)
        if number >= 6:
            continue
        off = P.pdlp(inst, eps=EPS, primal_weight=False)
        assert off.status == "optimal" and set(off.omega_path) == {1.0}            # Primalgewicht 1, wenn der Baustein aus ist
        norest = P.pdlp(inst, eps=EPS, restart=False, max_iter=300)
        assert norest.restarts == 0 and norest.restart_iters == [] and len(set(norest.omega_path)) == 1


# --- 7. Randfälle ----------------------------------------------------------------------------------------------------------------------------------------


def test_edge_cases_empty_single_variable_zero_objective_duplicate_rows():
    assert P.pdlp(_inst([], [], [1, 1], []), eps=EPS).status == "numerical"                      # keine Bedingungen: kein Ergebnis, nie "optimal"
    one = P.pdlp(_inst([[1]], [3], [1], [LE]), eps=EPS)
    assert one.status == "optimal" and one.obj == pytest.approx(3.0, abs=1e-4) and one.x == pytest.approx((3.0,), abs=1e-4)
    zero = P.pdlp(_inst([[1, 1]], [3], [0, 0], [LE]), eps=EPS)
    assert zero.status == "optimal" and abs(zero.obj) <= 1e-9 and _violation(_inst([[1, 1]], [3], [0, 0], [LE]), zero.x) <= 1e-3
    dup = P.pdlp(_inst([[1, 1], [1, 1], [2, 2]], [3, 3, 6], [1, 2], [LE, LE, LE]), eps=EPS)    # abhängige Zeilen: PDHG braucht keinen vollen Rang
    assert dup.status == "optimal" and dup.obj == pytest.approx(6.0, abs=1e-3)
    nothing = P.pdlp(_inst([[1, 1]], [0], [-1, -1], [LE]), eps=EPS)
    assert nothing.status == "optimal" and abs(nothing.obj) <= 1e-9
    eq = P.pdlp(_inst([[1, 1]], [2], [1, 3], [EQ]), eps=EPS)
    assert eq.status == "optimal" and eq.obj == pytest.approx(6.0, abs=1e-3)


def test_infeasible_and_unbounded_are_never_reported_optimal_under_scaling_and_renumbering():
    rng = np.random.default_rng(15)
    for base_inst in (S.infeasible_instance(), S.unbounded_instance()):
        A_, b, c, senses = _arrays(base_inst)
        for _ in range(2):
            f = float(rng.choice([0.1, 3.0, 25.0]))
            perm = rng.permutation(A_.shape[0])
            for v in (_inst(A_, b, f * c, senses), _inst(A_[perm], b[perm], c, [senses[i] for i in perm]), _inst(A_ * f, b * f, c, senses)):
                r = P.pdlp(v, eps=EPS)
                assert r.status != "optimal", (base_inst.kind, f)
                assert r.status in ("infeasible", "unbounded", "limit")
                if base_inst.kind == "infeasible":
                    assert r.status != "unbounded"
                else:
                    assert r.status != "infeasible"
    assert all(P.pdlp(v, eps=EPS).status == s for v, s in ((S.infeasible_instance(), "infeasible"), (S.unbounded_instance(), "unbounded")))


def test_preconditioning_is_a_positive_diagonal_scaling_that_keeps_the_zero_pattern():
    rng = np.random.default_rng(16)
    for _ in range(20):
        M = rng.uniform(0.1, 9.0, size=(int(rng.integers(2, 7)), int(rng.integers(3, 9))))
        M[rng.random(M.shape) < 0.3] = 0.0
        M[0, :] = np.where(M[0] == 0, 1.0, M[0])
        Ms, r, s = P.precondition(M)
        assert np.all(r > 0) and np.all(s > 0) and np.array_equal(Ms != 0, M != 0)
        assert np.allclose(Ms * r[:, None] * s, M, rtol=1e-12, atol=1e-12)


# --- 8. Determinismus ------------------------------------------------------------------------------------------------------------------------------------


def test_same_input_gives_the_same_output():
    for inst in _instances(4, offset=800):
        a, b = P.pdlp(inst, eps=EPS), P.pdlp(inst, eps=EPS)
        assert (a.status, a.iterations, a.matvecs, a.x, a.y, a.obj, a.restart_iters, a.omega_path) == (b.status, b.iterations, b.matvecs, b.x, b.y, b.obj, b.restart_iters, b.omega_path)
