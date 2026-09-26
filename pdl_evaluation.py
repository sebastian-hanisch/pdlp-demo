"""Auswertung von PDLP: ein Lauf gegen Grundform, Simplex und Innere Punkte, Ablation der fünf Bausteine, Genauigkeitsleiter und Ecke, Größe/Dichte im Operationsmodell, Skalierung, Klee-Minty-Würfel,
Erkennung von Unzulässigkeit und Unbeschränktheit."""

import statistics
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import pdl_algorithm as A
import pdl_constants as C
import pdl_ipm as IP
import pdl_pdlp as P
import pdl_scenario as S

RANDOM_KINDS = ("random", "mixed", "plateau")
ABLATION_CAP = 20000
FULL = {b: True for b in P.BLOCKS}
BASIC = {b: False for b in P.BLOCKS}


@dataclass(frozen=True)
class Settings:
    kind: str = "random"
    m: int = C.DEFAULT_M
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED
    density_i: int = C.DEFAULT_DENSITY_I
    eps_i: int = C.DEFAULT_EPS_I
    cap_i: int = C.DEFAULT_CAP_I
    scale_i: int = C.DEFAULT_SCALE_I
    average: bool = True
    restart: bool = True
    precond: bool = True
    adaptive: bool = True
    primal_weight: bool = True

    @property
    def density(self):
        return C.DENSITIES[min(max(self.density_i, 0), len(C.DENSITIES) - 1)]

    @property
    def eps(self):
        return 10.0 ** -C.EPS_EXPS[min(max(self.eps_i, 0), len(C.EPS_EXPS) - 1)]

    @property
    def cap(self):
        return C.CAPS[min(max(self.cap_i, 0), len(C.CAPS) - 1)]

    @property
    def scale_exp(self):
        return C.SCALE_EXPS[min(max(self.scale_i, 0), len(C.SCALE_EXPS) - 1)]

    @property
    def blocks(self):
        return {b: bool(getattr(self, b)) for b in P.BLOCKS}


def base_instance(s, seed=None):
    return S.generate(s.kind, s.m, s.n, s.density, s.seed if seed is None else seed)


def instance_of(s, seed=None):
    return S.column_scaled(base_instance(s, seed), s.scale_exp)


def family(s):
    """Instanzen für Sweeps: bei Zufalls-, Misch- und Plateau-Instanzen die fünf festen Seeds (Größe und Dichte aus dem Regler), sonst die eine gewählte Instanz."""
    if s.kind in RANDOM_KINDS:
        return [(sd, base_instance(s, sd), instance_of(s, sd)) for sd in C.SWEEP_SEEDS]
    return [(s.seed, base_instance(s), instance_of(s))]


def _med(values):
    return statistics.median(values) if values else float("nan")


def simplex_flops(inst, solution):
    """Operationsmodell des dichten Tableau-Simplex: je Pivot 2 (m+1)(Spalten+1)."""
    return solution.pivots * 2 * (inst.m + 1) * (A.standard_form(inst)[2]["ncols"] + 1)


def std_nnz(inst, x, tol=1e-9):
    """Nichtnullen einer Strukturlösung in der Standardform (Strukturvariablen plus Schlupf/Überschuss): eine Ecke hat höchstens m davon."""
    A_, b, _c = inst.arrays()
    x = np.array(x, dtype=float)
    slack = [b[i] - A_[i] @ x if s == S.LE else A_[i] @ x - b[i] for i, s in enumerate(inst.senses) if s != S.EQ]
    vec = np.concatenate([x, slack]) if slack else x
    return int(np.sum(np.abs(vec) > tol * max(1.0, float(np.max(np.abs(vec))))))


def rel_error(obj, ref):
    return abs(obj - ref) / (1.0 + abs(ref))


@dataclass
class Analysis:
    inst: object
    base: object
    res: object                      # PDLPResult der gewählten Bausteine
    basic: object                    # Grundform (alle Bausteine aus), gleiche Genauigkeit und Grenze
    full: object                     # PDLP mit allen fünf Bausteinen
    ipm: object                      # Mehrotra (Vergleich, Stück 8)
    simplex: object
    ref_status: str
    ref_obj: float
    error: float
    flops_pdlp: int
    flops_ipm: int
    flops_simplex: int


@lru_cache(maxsize=64)
def analyse(s):
    inst, base = instance_of(s), base_instance(s)
    keep = inst.n == 2
    res = P.pdlp(inst, eps=s.eps, max_iter=s.cap, keep=keep, **s.blocks)
    basic = res if s.blocks == BASIC else P.pdlp(inst, eps=s.eps, max_iter=s.cap, **BASIC)
    full = res if s.blocks == FULL else P.pdlp(inst, eps=s.eps, max_iter=s.cap, **FULL)
    ip = IP.ipm(inst, "mehrotra", eps=s.eps)
    sol, ref = A.solve(inst), A.solve(base)
    ref_obj = ref.obj if ref.status == "optimal" else float("nan")
    err = rel_error(res.obj, ref_obj) if res.x and ref.status == "optimal" else float("nan")
    return Analysis(inst, base, res, basic, full, ip, sol, ref.status, ref_obj, err, res.flops, ip.flops, simplex_flops(inst, sol) if sol.status == "optimal" else 0)


def block_configs():
    """Die zwölf Konfigurationen der Ablation: Grundform, jeder Baustein allein, PDLP voll, PDLP ohne je einen Baustein."""
    rows = [("Grundform", "ref", dict(BASIC))]
    for b in P.BLOCKS:
        rows.append((f"+ {P.BLOCK_LABELS[b]}", "alone", {**BASIC, b: True}))
    rows.append(("PDLP (alle fünf)", "ref", dict(FULL)))
    for b in P.BLOCKS:
        rows.append((f"ohne {P.BLOCK_LABELS[b]}", "removed", {**FULL, b: False}))
    return rows


@lru_cache(maxsize=32)
def ablation(s):
    """Iterationen und Matrix-Vektor-Produkte bis zur Genauigkeit für jede Konfiguration (Median über die feste Instanzfamilie); Läufe an der Grenze werden gezählt, nie als Ergebnis gemittelt."""
    rows = []
    fam = family(s)
    cap = min(s.cap, ABLATION_CAP)
    for label, kind, cfg in block_configs():
        its, mv, ok = [], [], 0
        for _sd, _base, inst in fam:
            r = P.pdlp(inst, eps=s.eps, max_iter=cap, **cfg)
            its.append(r.iterations), mv.append(r.matvecs)
            ok += r.status == "optimal"
        rows.append({"label": label, "kind": kind, "iterations": _med(its), "matvecs": _med(mv), "optimal": ok, "runs": len(fam), "cap": cap})
    return rows


@lru_cache(maxsize=32)
def eps_sweep(s):
    """Genauigkeitsleiter (gewählte Bausteine): Iterationen, Produkte, Nichtnullen der Lösung gegen die m Nichtnullen einer Ecke."""
    rows = []
    fam = family(s)
    for k in C.EPS_SWEEP_EXPS:
        its, mv, nz, sx, ok = [], [], [], [], 0
        for _sd, base, inst in fam:
            r = P.pdlp(inst, eps=10.0 ** -k, max_iter=s.cap, **s.blocks)
            its.append(r.iterations), mv.append(r.matvecs)
            ok += r.status == "optimal"
            if r.status == "optimal":
                nz.append(std_nnz(inst, r.x, tol=P.NNZ_TOL))
                sol = A.solve(inst)
                sx.append(std_nnz(inst, sol.x))
        rows.append({"k": k, "iterations": _med(its), "matvecs": _med(mv), "nnz_x": _med(nz), "nnz_vertex": _med(sx), "m": fam[0][2].m, "optimal": ok, "runs": len(fam)})
    return rows


@lru_cache(maxsize=32)
def size_sweep(s):
    """Zufallsinstanzen m = n (Dichte aus dem Regler): Iterationen und Operationen im Modell von PDLP (alle fünf Bausteine), Mehrotra und dichtem Tableau-Simplex; Kreuzungspunkte."""
    rows = []
    for n in C.SIZE_SIZES:
        it_p, it_i, pv, fl_p, fl_i, fl_s, ok = [], [], [], [], [], [], 0
        for sd in C.SIZE_SEEDS:
            inst = S.generate("random", n, n, s.density, sd)
            r = P.pdlp(inst, eps=s.eps, max_iter=50000)
            ip = IP.ipm(inst, "mehrotra", eps=s.eps)
            sol = A.solve(inst)
            ok += r.status == "optimal"
            it_p.append(r.iterations), it_i.append(ip.iterations), pv.append(sol.pivots), fl_p.append(r.flops), fl_i.append(ip.flops), fl_s.append(simplex_flops(inst, sol))
        rows.append({"n": n, "it_pdlp": _med(it_p), "it_ipm": _med(it_i), "pivots": _med(pv), "flops_pdlp": _med(fl_p), "flops_ipm": _med(fl_i), "flops_simplex": _med(fl_s), "optimal": ok,
                     "ratio_ipm": _med(fl_p) / max(_med(fl_i), 1.0), "ratio_simplex": _med(fl_p) / max(_med(fl_s), 1.0)})
    return {"rows": rows, "cross_ipm": _crossing(rows, "flops_ipm"), "cross_simplex": _crossing(rows, "flops_simplex")}


def _crossing(rows, key):
    for r in rows:
        if all(q["flops_pdlp"] < q[key] for q in rows if q["n"] >= r["n"]):
            return r["n"]
    return None


@lru_cache(maxsize=32)
def large_sparse(s):
    """Große dünne Zufallsinstanzen (Dichte 2 %): PDLP gegen dichten Tableau-Simplex im Operationsmodell (Mehrotra entfällt: die dichte Cholesky-Zerlegung braucht hier Minuten)."""
    rows = []
    for n in C.LARGE_SIZES:
        inst = S.generate("random", n, n, C.LARGE_DENSITY, C.SWEEP_SEEDS[0])
        r = P.pdlp(inst, eps=s.eps, max_iter=50000)
        sol = A.solve(inst)
        fs = simplex_flops(inst, sol)
        rows.append({"n": n, "status": r.status, "iterations": r.iterations, "pivots": sol.pivots, "flops_pdlp": r.flops, "flops_simplex": fs, "ratio": r.flops / max(fs, 1), "nnz": r.nnz_A})
    return rows


@lru_cache(maxsize=32)
def scale_sweep(s):
    """Schlechte Skalierung: dieselben fünf Zufallsinstanzen (10 × 10) mit Spalten über 10^k gestreut; Grundform, PDLP ohne Vorkonditionierung, PDLP: richtig gelöst (Optimum stimmt auf 0.1 %) und Iterationen."""
    cfgs = (("Grundform", dict(BASIC)), ("PDLP ohne Vorkonditionierung", {**FULL, "precond": False}), ("PDLP (alle fünf)", dict(FULL)))
    cap = min(s.cap, ABLATION_CAP)
    rows = []
    for k in C.SCALE_SWEEP_EXPS:
        row = {"k": k}
        for label, cfg in cfgs:
            right, its = 0, []
            for sd in C.SWEEP_SEEDS:
                base = S.generate("random", C.SCALE_SWEEP_SIZE, C.SCALE_SWEEP_SIZE, s.density, sd)
                true = A.solve(base).obj
                r = P.pdlp(S.column_scaled(base, k), eps=1e-6, max_iter=cap, **cfg)
                if r.status == "optimal" and rel_error(r.obj, true) <= 1e-3:
                    right += 1
                    its.append(r.iterations)
            row[label] = {"right": right, "iterations": _med(its)}
        rows.append(row)
    return {"rows": rows, "labels": [c[0] for c in cfgs]}


@lru_cache(maxsize=32)
def cube_sweep(s):
    """Klee-Minty-Würfel n = 2..14: Iterationen von PDLP (alle fünf Bausteine) und Mehrotra gegen die Pivots des Simplex (2^n − 1)."""
    rows = []
    for n in C.CUBE_SIZES:
        inst = S.klee_minty_instance(n)
        r = P.pdlp(inst, eps=s.eps, max_iter=50000)
        ip = IP.ipm(inst, "mehrotra", eps=s.eps)
        sol = A.solve(inst)
        rows.append({"n": n, "it_pdlp": r.iterations, "it_ipm": ip.iterations, "pivots": sol.pivots, "status": r.status, "flops_pdlp": r.flops, "flops_ipm": ip.flops, "flops_simplex": simplex_flops(inst, sol)})
    return rows


def constructed_infeasible(seed):
    """Zufallsinstanz mit dem Widerspruch x1 <= 4 und x1 >= 6."""
    base = S.generate("random", 6, 6, 0.6, seed)
    rows = [list(r) for r in base.A] + [[1.0] + [0.0] * 5, [1.0] + [0.0] * 5]
    return S.Instance(tuple(tuple(r) for r in rows), tuple(list(base.b) + [4.0, 6.0]), base.c, tuple(list(base.senses) + [S.LE, S.GE]), base.names, tuple(list(base.row_names) + ["Obergrenze", "Mindestmenge"]), "custom")


def constructed_unbounded(seed):
    """Zufallsinstanz, in der Dienst 3 keine Ressource verbraucht (Deckungsbeitrag positiv: wächst ohne Grenze)."""
    base = S.generate("random", 6, 6, 0.6, seed)
    rows = [list(r) for r in base.A]
    for r in rows:
        r[2] = 0.0
    return S.Instance(tuple(tuple(r) for r in rows), base.b, base.c, base.senses, base.names, base.row_names, "custom")


@lru_cache(maxsize=32)
def detection(s):
    """30 konstruierte unzulässige und 30 konstruierte unbeschränkte Instanzen: wie oft findet PDLP den nachgerechneten Strahl (Beweis), wie oft bleibt es an der Grenze; nie ein Optimum."""
    cap = min(s.cap, ABLATION_CAP)
    out = {}
    for name, maker in (("infeasible", constructed_infeasible), ("unbounded", constructed_unbounded)):
        found, limit, other, its = 0, 0, 0, []
        for sd in C.DETECT_SEEDS:
            r = P.pdlp(maker(sd), eps=1e-6, max_iter=cap)
            if r.status == name:
                found += 1
                its.append(r.iterations)
            elif r.status == "limit":
                limit += 1
            else:
                other += 1
        out[name] = {"found": found, "limit": limit, "other": other, "iterations": _med(its), "runs": len(C.DETECT_SEEDS)}
    return out
