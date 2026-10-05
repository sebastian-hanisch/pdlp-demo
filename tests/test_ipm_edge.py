"""Randfälle des Innere-Punkte-Kerns: Startpunkt schon optimal, Komplementaritätsmaß mu = 0 (Mini-Instanz min c^T z, z1 + z2 = 1, z >= 0)."""

import numpy as np
import pytest

import pdl_ipm as I

M = np.array([[1.0, 1.0]])
B = np.array([1.0])


def _run(method, c, start):
    return I.ipm(None, method=method, std=(M, B, np.array(c, dtype=float), 2, ""), start=start)


@pytest.mark.parametrize("method", I.METHODS)
def test_start_already_optimal_returns_the_point(method):
    # strikt positiv, Residuen ~1e-12, Lücke ~2e-12: schon im ersten Durchlauf optimal; vorher blieb x leer
    start = ([1.0, 1e-12], [-1e-12], [1e-12, 1.0 + 1e-12])
    res = _run(method, [0.0, 1.0], start)
    assert res.status == "optimal" and res.iterations == 0
    assert res.x == pytest.approx((1.0, 0.0), abs=1e-9) and res.y == pytest.approx((-1e-12,), abs=1e-9)
    assert res.obj == pytest.approx(-1e-12, abs=1e-9)


@pytest.mark.parametrize("method", I.METHODS)
def test_mu_zero_by_underflow_is_a_clean_numerical_stop(method):
    # z, s strikt positiv und endlich, aber z s = 0 durch Unterlauf, Residuen groß: vorher ZeroDivisionError (mehrotra) bzw. Unsinn
    res = _run(method, [1.0, 1.0], ([1e-200, 1e-200], [0.0], [1e-200, 1e-200]))
    assert res.status == "numerical" and res.x == ()


@pytest.mark.parametrize("method", I.METHODS)
def test_start_on_the_boundary_with_zero_complementarity(method):
    # z s = 0 (Nichtinnenpunkt): optimal, wenn das Zertifikat stimmt, sonst sauberer numerischer Abbruch (vorher ZeroDivisionError bei mehrotra)
    res = _run(method, [0.0, 1.0], ([1.0, 0.0], [0.0], [0.0, 1.0]))
    assert res.status == "optimal" and res.x == pytest.approx((1.0, 0.0))
    res = _run(method, [0.0, 2.0], ([1.0, 0.0], [0.0], [0.0, 1.0]))
    assert res.status == "numerical" and res.x == ()
