"""Literatur-Referenzwerte für PDLP (ORACLE-PLAYBOOK §5.1). Jeder Wert steht in einer Quelle, die beim Schreiben dieser Datei tatsächlich gelesen wurde (Stand der Einsicht: 2026-10-17);
nichts davon wurde mit dem Code der Demo erzeugt.

A. Applegate, Díaz, Hinder, Lu, Lubin, O'Donoghue, Schudy: *Practical Large-Scale Linear Programming using Primal-Dual Hybrid Gradient*, arXiv:2106.04756 (v1 vom 9. Juni 2021 und v2 vom 7. Januar 2022,
   beide gelesen; NeurIPS 2021), https://arxiv.org/abs/2106.04756. Fundstellen (Seitenzahlen der v2):
   - Gl. (3), (4), Abschnitt 2, S. 3 f.: PDHG-Schritt x+ = proj(x - tau (c - K^T y)), y+ = proj(y + sigma (q - K (2x+ - x))) mit tau = eta / omega, sigma = eta omega;
     Grundform: "set eta = 0.9 / ||K||_2 where ||K||_2 is estimated via power iteration, and for the primal weight we set omega = 1" (S. 4).
   - Algorithmus 1 (S. 4): Anfangsschrittweite eta_{0,0} = 1 / ||K||_inf (größter Betrag der Einträge), Anfangs-Primalgewicht InitializePrimalWeight(c, q).
   - Algorithmus 2 und Abschnitt 3.1 (S. 5), Gl. (5): eta_bar = ||z' - z||_omega^2 / (2 (y' - y)^T K (x' - x)); eta' = min((1 - (k+1)^-0.3) eta_bar, (1 + (k+1)^-0.6) eta);
     der Schritt wird angenommen, wenn eta <= eta_bar ("if eta <= eta_bar then return"). Die Normexponenten 0.3 und 0.6 bestätigt die Referenzsoftware (Punkt B).
   - Abschnitt 3.3 (S. 6): InitializePrimalWeight(c, q) = ||c||_2 / ||q||_2, wenn ||c||_2 und ||q||_2 größer als eine kleine Toleranz sind, sonst 1;
     Algorithmus 3: PrimalWeightUpdate = exp(theta log(Delta_y / Delta_x) + (1 - theta) log(omega_{n-1})), wenn Delta_x und Delta_y größer als die Toleranz sind, sonst omega_{n-1};
     "In PDLP, we use theta = 0.5." Das Gewicht wird nur nach einem Neustart aktualisiert.
   - Abschnitt 3.5 (S. 7): Ruiz-Skalierung "the infinity norm of each row and each column converge to 1"; Standardeinstellung 10 Ruiz-Schritte, danach Pock-Chambolle.
   - Abschnitt 4.1, Gl. (6) (S. 8): Abbruch bei |Dualziel - Primalziel| <= eps (1 + |Dualziel| + |Primalziel|), ||Ax - b||_2 <= eps (1 + ||q||_2), ||c - K^T y - lambda||_2 <= eps (1 + ||c||_2).
   - Anhang A (S. 15), Proposition 1: Skaleninvarianz der Grundform (siehe test_eigenschaften_pdlp.py).
   Nicht übernommen (Abweichung der Demo, kein Test): Die Arbeit nennt für die Neustart-Kriterien beta_sufficient = 0.9, beta_necessary = 0.1, beta_artificial = 0.5 (S. 6, Anhang C.2), im Text mit
   vertauschten Namen gegenüber der Referenzsoftware (dort sufficient 0.1, necessary 0.9, Punkt B), und das Kriterium arbeitet mit der normalisierten Dualitätslücke. Die Demo nutzt den KKT-Fehler und
   die Werte 0.2 / 0.8 / 0.36 (pdl_pdlp.py). Das ist eine bewusste Vereinfachung der Demo, kein Test dazu.
B. Referenzsoftware OR-Tools PDLP, solvers.proto, https://raw.githubusercontent.com/google/or-tools/stable/ortools/pdlp/solvers.proto, eingesehen 2026-10-17:
   primal_weight_update_smoothing = 0.5; step_size_reduction_exponent = 0.3; step_size_growth_exponent = 0.6; linesearch_rule = ADAPTIVE_LINESEARCH_RULE; Abbruchtoleranzen 1e-6 (veraltete Felder
   eps_optimal_absolute / eps_optimal_relative).
C. Hillier & Lieberman, Introduction to Operations Research, Kap. 4, https://people.ohio.edu/melkonia/math4620/chapter4.pdf, eingesehen 2026-10-17: Wyndor-Beispiel Optimum (2, 6), Z = 36
   (Abschn. 4.3 S. 123, Tabelle 4.8 S. 128); Strahlentherapie-Beispiel (Abschn. 4.6 S. 138, Abb. 4.5): min 0.4 x1 + 0.5 x2 unter 0.3 x1 + 0.1 x2 <= 2.7, 0.5 x1 + 0.5 x2 = 6, 0.6 x1 + 0.4 x2 >= 6,
   Optimum (7.5, 4.5), Z = 5.25 (Relationszeichen in der Textextraktion verloren, aus dem Zusammenhang der Quelle gelesen).
D. Wikipedia-Beispiel (gegengerechnet), "Revised simplex method", Abschnitt "Numerical example", eingesehen 2026-10-17: min -2x -3y -4z unter 3x + 2y + z <= 10, 2x + 5y + 3z <= 15;
   Optimum x = (0, 0, 5) (Schlupf 5, 0), Zielwert -20 (von Hand aus c^T x gerechnet; der Artikel nennt den Wert nicht). Die Demo maximiert, also Zielwert +20.

Handrechnungen: die Zahlen in den Formeltests (Primalgewicht 4, Obergrenze der Schrittweite) sind von Hand aus den Formeln der Arbeit gerechnet, nicht aus dem Code.

Nicht eingesehen / nicht eindeutig (kein Test, siehe Bericht "offen"): Pock-Chambolle-Exponent der Skalierung (das Wurzelzeichen ist in der Textextraktion der Arbeit verloren), Chambolle & Pock (2011),
Lu & Yang (2023), Tabellen 1 bis 4 der Arbeit (MIPLIB-Instanzen, nicht reproduzierbar)."""

import math

import numpy as np
import pytest

import pdl_pdlp as P
import pdl_scenario as S

LE, GE, EQ = S.LE, S.GE, S.EQ


def _inst(rows, b, c, senses):
    n = len(c)
    return S.Instance(tuple(tuple(float(v) for v in r) for r in rows), tuple(float(v) for v in b), tuple(float(v) for v in c), tuple(senses),
                      tuple(f"x{j}" for j in range(n)), tuple(f"r{i}" for i in range(len(b))), "custom")


RADIATION = _inst([[0.3, 0.1], [0.5, 0.5], [0.6, 0.4]], [2.7, 6.0, 6.0], [-0.4, -0.5], [LE, EQ, GE])      # max -Z
WIKIPEDIA = _inst([[3, 2, 1], [2, 5, 3]], [10, 15], [2, 3, 4], [LE, LE])                                  # max 2x + 3y + 4z


# --- Zahlenbeispiele aus Lehrbuch und Wikipedia (C, D) ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kw", [{}, {"precond": False}])
def test_textbook_examples_give_the_printed_optima(kw):
    r = P.pdlp(S.textbook_instance(), eps=1e-7, **kw)                                                         # Wyndor: (2, 6), Z = 36
    assert r.status == "optimal" and r.obj == pytest.approx(36.0, abs=1e-4) and r.x == pytest.approx((2.0, 6.0), abs=1e-4)
    r = P.pdlp(RADIATION, eps=1e-7, **kw)                                                                     # Strahlentherapie: (7.5, 4.5), Z = 5.25
    assert r.status == "optimal" and -r.obj == pytest.approx(5.25, abs=1e-4) and r.x == pytest.approx((7.5, 4.5), abs=1e-3)
    r = P.pdlp(WIKIPEDIA, eps=1e-7, **kw)                                                                     # Wikipedia: (0, 0, 5), min-Wert -20
    assert r.status == "optimal" and r.obj == pytest.approx(20.0, abs=1e-4) and r.x == pytest.approx((0.0, 0.0, 5.0), abs=1e-3)


# --- Grundform: Schrittweite 0.9 / ||K||_2, Primalgewicht 1 (A, S. 4) -----------------------------------------------------------------------------------


def test_baseline_pdhg_uses_step_size_point_nine_over_the_spectral_norm_and_primal_weight_one():
    for inst in (S.textbook_instance(), S.centre_instance(), S.generate("random", 6, 8, 0.6, 1), RADIATION):
        r = P.pdlp(inst, eps=1e-6, max_iter=6, average=False, restart=False, precond=False, adaptive=False, primal_weight=False)
        M = P.pdlp_form(inst)[0]
        sigma_max = float(np.linalg.norm(M, 2))                                                            # unabhängig mit LAPACK gerechnet
        assert r.norm_est == pytest.approx(sigma_max, rel=1e-6)                                            # Potenzmethode trifft die Spektralnorm
        assert r.eta_path == pytest.approx([0.9 / sigma_max] * len(r.eta_path), rel=1e-6)
        assert set(r.omega_path) == {1.0}


# --- Anfangs-Primalgewicht ||c|| / ||q|| (A, Abschnitt 3.3) ----------------------------------------------------------------------------------------------


def test_initial_primal_weight_is_the_norm_ratio_of_c_and_q_and_one_otherwise():
    for inst in (S.textbook_instance(), S.centre_instance(), S.generate("random", 6, 8, 0.6, 2), S.generate("mixed", 7, 7, 0.6, 3), RADIATION):
        A_, b, c = inst.arrays()
        r = P.pdlp(inst, eps=1e-6, max_iter=4, precond=False, adaptive=False, average=False, restart=False, primal_weight=True)
        assert r.omega_path[0] == pytest.approx(np.linalg.norm(c) / np.linalg.norm(b), rel=1e-12)      # Schlupfspalten haben Kosten 0: ||c~|| = ||c||
    zero_rhs = _inst([[1, 1], [1, 2]], [0, 0], [-1, -2], [LE, LE])                                         # ||q|| = 0 -> 1
    zero_cost = _inst([[1, 1], [1, 2]], [3, 4], [0, 0], [LE, LE])                                          # ||c|| = 0 -> 1
    for inst in (zero_rhs, zero_cost):
        r = P.pdlp(inst, eps=1e-6, max_iter=4, precond=False, adaptive=False, average=False, restart=False, primal_weight=True)
        assert r.omega_path[0] == 1.0


# --- Algorithmus 3: Aktualisierung des Primalgewichts (A, S. 6) -------------------------------------------------------------------------------------------


def test_primal_weight_update_matches_algorithm_three_by_hand():
    assert P.PW_SMOOTHING == 0.5                                                                           # "In PDLP, we use theta = 0.5." (S. 6; OR-Tools: 0.5)
    # omega = 2, Delta_x = 1, Delta_y = 8: exp(0.5 ln 8 + 0.5 ln 2) = sqrt(16) = 4 (von Hand)
    assert P.primal_weight_update(2.0, 1.0, 8.0) == pytest.approx(4.0, rel=1e-12)
    # omega = 5, Delta_x = 4, Delta_y = 1: exp(0.5 ln(1/4) + 0.5 ln 5) = sqrt(5/4)
    assert P.primal_weight_update(5.0, 4.0, 1.0) == pytest.approx(math.sqrt(1.25), rel=1e-12)
    assert P.primal_weight_update(3.0, 0.0, 2.0) == 3.0 and P.primal_weight_update(3.0, 2.0, 0.0) == 3.0  # keine Bewegung: unverändert
    for w in (0.1, 1.0, 7.0):                                                                              # Delta_y / Delta_x = omega: Fixpunkt
        assert P.primal_weight_update(w, 2.0, 2.0 * w) == pytest.approx(w, rel=1e-12)


# --- Algorithmus 2 und Gl. (5): adaptive Schrittweite (A, S. 5; B: Exponenten 0.3 und 0.6) -----------------------------------------------------------------


def test_adaptive_step_size_rule_matches_algorithm_two_by_hand():
    # k = 3, omega = 2, dx = (1, 0), dy = (0, 2), K dx = (0, 3): ||dz||_omega^2 = 2 * 1 + 4 / 2 = 4, |dy^T K dx| = 6, eta_bar = 4 / (2 * 6) = 1/3 (von Hand)
    cap = P.adaptive_cap(3, 2.0, np.array([1.0, 0.0]), np.array([0.0, 2.0]), np.array([0.0, 3.0]))
    assert cap == pytest.approx((1.0 - 4.0 ** -0.3) / 3.0, rel=1e-12)
    assert P.next_eta(3, 0.5, cap) == pytest.approx(min(cap, (1.0 + 4.0 ** -0.6) * 0.5), rel=1e-12)
    assert P.next_eta(3, 0.1, 1e9) == pytest.approx((1.0 + 4.0 ** -0.6) * 0.1, rel=1e-12)               # Wachstumsbegrenzung greift, wenn die Obergrenze groß ist
    assert P.adaptive_cap(3, 1.0, np.array([1.0]), np.array([1.0]), np.array([0.0])) == math.inf        # kein gemischter Term: keine Grenze
    # der Schritt wird nur angenommen, wenn eta <= eta_bar: im Lauf gilt das für jede akzeptierte Iteration
    r = P.pdlp(S.generate("random", 6, 8, 0.6, 4), eps=1e-5, keep=True, precond=False, restart=False)
    assert r.trace and all(eta <= cap * (1.0 + 1e-12) for eta, cap in r.trace)


def test_first_step_size_is_at_most_one_over_the_largest_entry():
    for inst in (S.textbook_instance(), S.centre_instance(), S.generate("random", 6, 8, 0.6, 5)):
        M = P.pdlp_form(inst)[0]
        r = P.pdlp(inst, eps=1e-6, max_iter=1, precond=False, keep=True)
        assert r.trace[0][0] <= 1.0 / np.abs(M).max() * (1.0 + 1e-12)                                      # eta_{0,0} = 1 / ||K||_inf, nur verkleinert (S. 4 und 5)


# --- Ruiz-Skalierung (A, Abschnitt 3.5) ------------------------------------------------------------------------------------------------------------------


def test_ruiz_scaling_drives_row_and_column_infinity_norms_to_one():
    rng = np.random.default_rng(5)
    for _ in range(15):
        M = rng.uniform(0.05, 50.0, size=(int(rng.integers(3, 8)), int(rng.integers(4, 10)))) * 10.0 ** rng.uniform(-3, 3, size=(1, 1))
        M *= 10.0 ** rng.uniform(-2, 2, size=(M.shape[0], 1))
        M[rng.random(M.shape) < 0.3] = 0.0
        M[:, np.abs(M).max(axis=0) == 0] = 1.0
        M[np.abs(M).max(axis=1) == 0, :] = 1.0
        for sweeps, tol in ((10, 5e-2), (200, 1e-9)):
            Ms, _r, _s = P.precondition(M, ruiz=sweeps, pock_chambolle=False)
            assert np.abs(np.abs(Ms).max(axis=1) - 1.0).max() < tol and np.abs(np.abs(Ms).max(axis=0) - 1.0).max() < tol


# --- Abbruchkriterien (6) (A, S. 8) -----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("eps", [1e-4, 1e-6])
def test_result_satisfies_the_published_optimality_criteria(eps):
    for inst in (S.textbook_instance(), RADIATION, WIKIPEDIA, S.centre_instance()):
        r = P.pdlp(inst, eps=eps)
        assert r.status == "optimal"
        M, q, c, _n, _ = P.pdlp_form(inst)                                                                 # min c^T x, M x = q, x >= 0
        x, y = np.array(r.x_full), np.array(r.y)
        lam = np.maximum(c - M.T @ y, 0.0)                                                                 # lambda = proj(c - K^T y) für Schranken [0, inf)
        dual_obj, primal_obj = float(q @ y), float(c @ x)
        assert abs(dual_obj - primal_obj) <= eps * (1.0 + abs(dual_obj) + abs(primal_obj))                  # (6a)
        assert np.linalg.norm(M @ x - q) <= eps * (1.0 + np.linalg.norm(q))                                 # (6b)
        assert np.linalg.norm(c - M.T @ y - lam) <= eps * (1.0 + np.linalg.norm(c)) * (1.0 + 1e-12)        # (6c)
