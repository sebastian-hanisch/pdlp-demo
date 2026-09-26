"""Instanzen: Bauart, Determinismus, Fixtures (Lehrbuch, Zentrum, entartete Ecke), Zulässigkeit der erzeugten Instanzen, Zentrum."""

import numpy as np
import pytest
from scipy.optimize import linprog

import pdl_scenario as S


def _highs(inst, zero_objective=False):
    A, b, c = inst.arrays()
    if zero_objective:
        c = np.zeros_like(c)
    ub_a, ub_b, eq_a, eq_b = [], [], [], []
    for i, s in enumerate(inst.senses):
        if s == S.LE:
            ub_a.append(A[i]), ub_b.append(b[i])
        elif s == S.GE:
            ub_a.append(-A[i]), ub_b.append(-b[i])
        else:
            eq_a.append(A[i]), eq_b.append(b[i])
    return linprog(-c, A_ub=np.array(ub_a) if ub_a else None, b_ub=ub_b or None, A_eq=np.array(eq_a) if eq_a else None, b_eq=eq_b or None, bounds=(0, None), method="highs")


def reference_status(inst):
    """Status laut HiGHS; bei zulässigen, unbeschränkten LPs meldet HiGHS gelegentlich "unzulässig" (Präsolve), darum wird über ein Zulässigkeitsproblem ohne Zielfunktion abgesichert."""
    h = _highs(inst)
    if h.status == 0:
        return "optimal"
    return "infeasible" if _highs(inst, zero_objective=True).status == 2 else "unbounded"


def test_textbook_is_the_classic_two_service_example():
    inst = S.textbook_instance()
    assert (inst.m, inst.n) == (3, 2) and inst.senses == (S.LE,) * 3 and inst.b == (4.0, 12.0, 18.0) and inst.c == (3.0, 5.0)
    assert -_highs(inst).fun == pytest.approx(36.0)


def test_fixtures_have_the_status_they_claim():
    assert _highs(S.infeasible_instance()).status == 2
    assert _highs(S.unbounded_instance()).status == 3
    d = S.degenerate_instance()
    assert -_highs(d).fun == pytest.approx(36.0) and d.A[3] == (1.0, 1.0) and d.b[3] == 8.0 and d.kind == "degenerate"


def test_centre_instance_is_the_constructed_one():
    inst = S.centre_instance()
    assert (inst.m, inst.n) == (4, 5) and inst.senses == (S.LE,) * 4 and inst.b == (80.0, 200.0, 55.0, 400.0) and inst.c == (15.5, 15.0, 26.0, 20.0, 12.0)
    h = _highs(inst)
    assert -h.fun == pytest.approx(720.0) and list(h.x) == pytest.approx([20, 10, 10, 0, 0])


def test_generation_is_deterministic_and_seed_dependent_and_hashable():
    a = S.generate("random", 8, 6, 0.5, 3)
    assert a == S.generate("random", 8, 6, 0.5, 3) and a != S.generate("random", 8, 6, 0.5, 4) and hash(a) == hash(S.generate("random", 8, 6, 0.5, 3))


@pytest.mark.parametrize("kind", ["random", "mixed"])
def test_generated_instances_are_feasible_and_bounded(kind):
    for seed in range(40):
        for m, n in ((2, 2), (6, 4), (4, 9), (15, 15), (10, 80)):
            inst = S.generate(kind, m, n, 0.1, seed)
            assert (inst.m, inst.n) == (m, n) and len(inst.senses) == m and _highs(inst).status == 0, (kind, seed, m, n)


def test_every_row_and_column_is_used_and_row_zero_is_dense_even_at_low_density():
    for seed in range(20):
        inst = S.generate("random", 10, 40, 0.02, seed)
        A = np.array(inst.A)
        assert (A.sum(axis=1) > 0).all() and (A.sum(axis=0) > 0).all() and (A[0] > 0).all() and (A >= 0).all()


def test_mixed_instances_contain_all_three_senses_sometimes():
    seen = set()
    for seed in range(30):
        seen |= set(S.generate("mixed", 12, 8, 0.5, seed).senses)
    assert seen == {S.LE, S.GE, S.EQ}
    for seed in range(30):
        inst = S.generate("mixed", 10, 4, 0.5, seed)
        assert sum(s == S.EQ for s in inst.senses) <= 2 and inst.senses[0] == S.LE


def test_kinds_and_fixtures_are_consistent_and_unknown_kind_raises():
    assert set(S.FIXTURE_KINDS) <= set(S.KINDS) and set(S.KIND_LABELS) == set(S.KINDS)
    for kind in S.FIXTURE_KINDS:
        assert S.generate(kind, 9, 9, 0.5, 9) == S.generate(kind, 3, 3, 0.1, 1)
    with pytest.raises(ValueError):
        S.generate("nope", 3, 3, 0.5, 1)


def test_klee_minty_cube_has_the_chvatal_form_and_the_known_optimum():
    inst = S.klee_minty_instance(3)
    assert inst.A == ((1.0, 0.0, 0.0), (4.0, 1.0, 0.0), (8.0, 4.0, 1.0)) and inst.b == (5.0, 25.0, 125.0) and inst.c == (4.0, 2.0, 1.0)
    for n in (2, 5, 9, 14):
        inst = S.klee_minty_instance(n)
        assert (inst.m, inst.n) == (n, n) and _highs(inst).status == 0 and -_highs(inst).fun == pytest.approx(5.0 ** n, rel=1e-9)
    assert S.generate("klee_minty", 99, 7, 0.5, 3) == S.klee_minty_instance(7) and S.klee_minty_instance(50).n == S.CUBE_MAX and S.klee_minty_instance(1).n == 2
    assert "klee_minty" in S.KINDS and "klee_minty" in S.KIND_LABELS and "klee_minty" not in S.FIXTURE_KINDS
