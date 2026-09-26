"""Dichtes Tableau-Simplex (Kern aus den Stücken 1-4, Dantzig-Regel mit Bland-Notbremse gegen Zyklen) mit Endtableau: Basis, Duale y (Zeilenorientierung), reduzierte Kosten und B^-1 stehen im Endtableau."""

from dataclasses import dataclass, field

import numpy as np

import pdl_scenario as S

LE, GE, EQ = S.LE, S.GE, S.EQ
TOL = 1e-9
MAX_PIVOTS = 20_000
STALL_LIMIT = 25                            # so viele Nullschritte in Folge: bis zum nächsten echten Schritt gilt die Bland-Regel (gegen Zyklen)


def standard_form(inst):
    """Gleichungsform mit Schlupf-, Überschuss- und künstlichen Variablen; Zeilen mit negativem b werden mit -1 multipliziert (Sinn dreht sich). Gibt (T, basis, info) mit T = [A' | rhs] (m Zeilen).
    Die Anfangsbasis besteht aus Einheitsspalten (Schlupf bei <=, künstliche Variable sonst), B^-1 steht also im Endtableau unter diesen Spalten."""
    A, b, c = inst.arrays()
    m, n = A.shape
    rows, rhs, senses, sign = [], [], [], []
    for i in range(m):
        a, r, s = A[i].copy(), float(b[i]), inst.senses[i]
        sg = 1
        if r < 0:
            a, r, sg = -a, -r, -1
            s = {LE: GE, GE: LE, EQ: EQ}[s]
        rows.append(a), rhs.append(r), senses.append(s), sign.append(sg)
    slack_col, art_col = {}, {}
    ncols = n
    for i, s in enumerate(senses):
        if s in (LE, GE):
            slack_col[i] = ncols
            ncols += 1
    for i, s in enumerate(senses):
        if s in (GE, EQ):
            art_col[i] = ncols
            ncols += 1
    T = np.zeros((m, ncols + 1))
    basis = []
    for i in range(m):
        T[i, :n] = rows[i]
        T[i, -1] = rhs[i]
        if i in slack_col:
            T[i, slack_col[i]] = 1.0 if senses[i] == LE else -1.0
        if i in art_col:
            T[i, art_col[i]] = 1.0
        basis.append(art_col[i] if i in art_col else slack_col[i])
    names = list(inst.names) + [None] * (ncols - n)
    for i, j in slack_col.items():
        names[j] = f"s{i + 1}" if senses[i] == LE else f"e{i + 1}"
    for i, j in art_col.items():
        names[j] = f"a{i + 1}"
    info = {"n": n, "m": m, "ncols": ncols, "slack_col": slack_col, "art_col": art_col, "sign": sign, "senses": senses, "names": names, "c": c, "b": b}
    return T, basis, info


def _pivot(T, row, col):
    T[row] /= T[row, col]
    factors = T[:, col].copy()
    factors[row] = 0.0
    idx = np.nonzero(factors)[0]
    if len(idx):
        T[idx] -= np.outer(factors[idx], T[row])


@dataclass
class Solution:
    status: str                              # "optimal" | "infeasible" | "unbounded" | "limit"
    x: tuple = ()
    obj: float = float("nan")
    y: tuple = ()                            # Duale je ursprünglicher Zeile: Änderung des Optimums je Einheit b_i
    pivots: int = 0
    basis: tuple = ()                        # Basisspalten des Endtableaus in Zeilenreihenfolge
    T: object = None                         # Endtableau (m + 1 Zeilen: Bedingungen und Zielzeile)
    info: dict = field(default_factory=dict)
    init_basis: tuple = ()                   # Spalten der Anfangsbasis (Einheitsmatrix)

    @property
    def basic_set(self):
        return frozenset(self.basis)


def solve(inst, max_pivots=MAX_PIVOTS):
    """Zwei-Phasen-Simplex; bei "optimal" enthält das Ergebnis das Endtableau für die Sensitivitätsanalyse."""
    T, basis, info = standard_form(inst)
    m, n, ncols = info["m"], info["n"], info["ncols"]
    art = set(info["art_col"].values())
    init_basis = tuple(basis)
    full = np.zeros((m + 1, ncols + 1))
    full[:m] = T
    T = full
    count = [0]

    def run(cvec, allowed):
        cB = cvec[basis]
        T[m, :] = cB @ T[:m, :]
        T[m, :-1] -= cvec
        stall = 0
        while True:
            r = T[m, :-1]
            cand = [j for j in allowed if r[j] < -TOL]
            if not cand:
                return "optimal"
            if stall >= STALL_LIMIT:
                enter = cand[0]
            else:
                best = min(r[j] for j in cand)
                enter = next(j for j in cand if r[j] <= best + 1e-9 * max(1.0, abs(best)))
            col = T[:m, enter]
            pos = [i for i in range(m) if col[i] > TOL]
            if not pos:
                return "unbounded"
            ratios = {i: T[i, -1] / col[i] for i in pos}
            rmin = min(ratios.values())
            tied = [i for i in pos if ratios[i] <= rmin + TOL * max(1.0, abs(rmin))]
            leave = min(tied, key=lambda i: basis[i])
            _pivot(T, leave, enter)
            basis[leave] = enter
            count[0] += 1
            stall = stall + 1 if max(rmin, 0.0) <= TOL else 0
            if count[0] > max_pivots:
                return "limit"

    if art:
        c1 = np.zeros(ncols)
        for j in art:
            c1[j] = -1.0
        st = run(c1, list(range(ncols)))
        if st == "limit":
            return Solution("limit", pivots=count[0], info=info)
        if T[m, -1] < -1e-7:
            return Solution("infeasible", pivots=count[0], info=info)
        for a in sorted(art):
            if a not in basis:
                continue
            i = basis.index(a)
            cand = [j for j in range(ncols) if j not in art and abs(T[i, j]) > TOL]
            if cand:
                best = max(abs(T[i, j]) for j in cand)
                enter = next(j for j in cand if abs(T[i, j]) >= best - 1e-9 * max(1.0, best))
                _pivot(T, i, enter)
                basis[i] = enter
                count[0] += 1
    c2 = np.zeros(ncols)
    c2[:n] = info["c"]
    st = run(c2, [j for j in range(ncols) if j not in art])
    if st != "optimal":
        return Solution(st, pivots=count[0], info=info)
    xf = np.zeros(ncols)
    for i, j in enumerate(basis):
        xf[j] = T[i, -1]
    y_std = [float(T[m, init_basis[i]]) for i in range(m)]
    y = tuple(y_std[i] * info["sign"][i] for i in range(m))
    return Solution("optimal", tuple(float(v) for v in xf[:n]), float(np.dot(info["c"], xf[:n])), y, count[0], tuple(basis), T, info, init_basis)


# --- Prüfgrößen (für Tests und Auswertung) ---------------------------------------------------------------------------------------------------------


def primal_violation(inst, x):
    """Größte Verletzung der Bedingungen und der Nichtnegativität durch x (0 = zulässig)."""
    A, b, _c = inst.arrays()
    x = np.asarray(x, dtype=float)
    worst = max(0.0, float(-x.min())) if len(x) else 0.0
    for i in range(inst.m):
        act, s = float(A[i] @ x), inst.senses[i]
        worst = max(worst, max(0.0, act - b[i]) if s == LE else (max(0.0, b[i] - act) if s == GE else abs(act - b[i])))
    return worst


def dual_violation(inst, y):
    """Größte Verletzung der Dual-Zulässigkeit (A^T y >= c, y >= 0 bei <=, y <= 0 bei >=)."""
    A, _b, c = inst.arrays()
    y = np.asarray(y, dtype=float)
    worst = float(max(0.0, (c - A.T @ y).max())) if inst.n else 0.0
    for i in range(inst.m):
        s = inst.senses[i]
        worst = max(worst, max(0.0, -y[i]) if s == LE else (max(0.0, y[i]) if s == GE else 0.0))
    return worst
