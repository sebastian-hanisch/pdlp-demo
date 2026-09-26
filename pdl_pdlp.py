"""PDLP (Applegate u. a. 2021): primal-duales Hybrid-Gradienten-Verfahren (PDHG, Chambolle & Pock 2011) für  max c·x,  A x (<=|>=|=) b,  x >= 0  ohne jede Faktorisierung.

Standardform: min c~^T x  unter  M x = b,  x >= 0  (Schlupf/Überschuss angehängt, c~ = -c). Sattelpunktform  min_(x>=0) max_y  c~^T x - y^T (M x - b). Eine Iteration:
    x+ = max(0, x - tau (c~ - M^T y)),   y+ = y + sigma (b - M (2 x+ - x)),   tau = eta / omega,  sigma = eta * omega,  eta <= 1 / ||M||  (Grundform),
je Iteration zwei Matrix-Vektor-Produkte (M x+ und M^T y+). Fünf Bausteine machen daraus PDLP (jeder einzeln ein- und ausschaltbar):
    average      laufender, mit der Schrittweite gewichteter Mittelwert seit dem letzten Neustart (der Mittelwert liefert die O(1/k)-Garantie),
    restart      Neustart auf den besseren von aktueller Iterierter und Mittelwert (KKT-Fehler: hinreichend 0.2, notwendig 0.8 ohne Fortschritt, künstlich 0.36),
    precond      Ruiz-Äquilibrierung (10 Sweeps, Max-Norm) und Pock-Chambolle (1-Norm, alpha = 1): x = x'/s, y = y'/r auf dem skalierten Problem M' = M / (r s^T),
    adaptive     adaptive Schrittweite (Formel aus PDLP: eta' = min((1 - (k+1)^-0.3) ||dz||_w^2 / (2 |dy^T M dx|), (1 + (k+1)^-0.6) eta), Wiederholung bis eta <= eta'),
    primal_weight  Primalgewicht omega = ||c|| / ||b||, bei jedem Neustart geglättet auf die Bewegungen: omega <- exp(0.5 ln(dy / dx) + 0.5 ln omega).
Abbruch: relative Residuen primal, dual und Lücke <= eps, ALLE am ORIGINAL nachgerechnet (aus den zwischengespeicherten Produkten des skalierten Problems: M x - b = r (M' x' - b'), M^T y - c = s (M'^T y' - c'))."""

import math
from dataclasses import dataclass, field

import numpy as np

import pdl_scenario as S
from pdl_ipm import dual_ray, primal_ray

LE, GE, EQ = S.LE, S.GE, S.EQ
CHECK_EVERY = 4                       # Auswertung von Abbruch und Neustart alle 4 Iterationen (die PDLP-Software: 64; hier klein, weil die Instanzen klein sind)
RAY_EVERY = 64                        # Strahltests (Farkas, unbeschränkt) alle 64 Iterationen (jeweils äquilibriert, O(m N))
MAX_ITER = 20000
NORM_ITERS = 40                       # Potenzmethode für ||M||_2 (feste Startrichtung, kein Zufall)
BETA_SUFFICIENT, BETA_NECESSARY, BETA_ARTIFICIAL = 0.2, 0.8, 0.36
PW_SMOOTHING = 0.5
RUIZ_SWEEPS = 10
NNZ_TOL = 1e-6                        # relativ zur größten Komponente: darunter gilt eine Komponente als Null (Zählung der Nichtnullen)
MAX_TRIALS = 60
REJECT_SHRINK = 0.9                   # nach einer Ablehnung wird nur 0.9 der Obergrenze versucht (die reine Obergrenze konvergiert sonst geometrisch gegen einen Fixpunkt und wird bis zur Rundungsgenauigkeit immer wieder knapp abgelehnt)
BLOCKS = ("average", "restart", "precond", "adaptive", "primal_weight")
BLOCK_LABELS = {"average": "Mittelung", "restart": "Neustarts", "precond": "Vorkonditionierung", "adaptive": "adaptive Schrittweite", "primal_weight": "Primalgewicht"}


def pdlp_form(inst):
    """Standardform ohne Rangprüfung: Matrix M (m x N), rechte Seite b, Kosten c (Minimierung), Zahl der Strukturvariablen n. PDHG braucht keine unabhängigen Zeilen."""
    A, b, c = inst.arrays()
    m, n = A.shape
    extra = [i for i, s in enumerate(inst.senses) if s != EQ]
    M = np.hstack([A, np.zeros((m, len(extra)))])
    for k, i in enumerate(extra):
        M[i, n + k] = 1.0 if inst.senses[i] == LE else -1.0
    return M, b, np.concatenate([-c, np.zeros(len(extra))]), n, ""


def precondition(M, ruiz=RUIZ_SWEEPS, pock_chambolle=True):
    """Diagonale Skalierung M' = diag(1/r) M diag(1/s): Ruiz (Max-Norm) und Pock-Chambolle (alpha = 1, 1-Norm). Rückgabe (M', r, s); Nullzeilen und Nullspalten bleiben unskaliert (Faktor 1)."""
    A = np.array(M, dtype=float)
    r, s = np.ones(A.shape[0]), np.ones(A.shape[1])
    for _ in range(ruiz):
        rr = np.sqrt(np.abs(A).max(axis=1)) if A.shape[1] else np.ones(A.shape[0])
        ss = np.sqrt(np.abs(A).max(axis=0)) if A.shape[0] else np.ones(A.shape[1])
        rr[rr == 0], ss[ss == 0] = 1.0, 1.0
        A = A / rr[:, None] / ss
        r, s = r * rr, s * ss
    if pock_chambolle:
        rr, ss = np.sqrt(np.abs(A).sum(axis=1)), np.sqrt(np.abs(A).sum(axis=0))
        rr[rr == 0], ss[ss == 0] = 1.0, 1.0
        A = A / rr[:, None] / ss
        r, s = r * rr, s * ss
    return A, r, s


class MatOp:
    """Matrix mit Zählern: jedes Produkt A x (mul) und A^T y (tmul) ist ein Matrix-Vektor-Produkt (2 nnz Operationen)."""

    def __init__(self, A):
        self.A = A
        self.mul_calls = 0
        self.tmul_calls = 0

    @property
    def calls(self):
        return self.mul_calls + self.tmul_calls

    def mul(self, x):
        self.mul_calls += 1
        return self.A @ x

    def tmul(self, y):
        self.tmul_calls += 1
        return self.A.T @ y

    def norm(self, iters=NORM_ITERS):
        """Schätzung von ||A||_2 (Potenzmethode auf A^T A, feste Startrichtung; jede Runde zwei Produkte, mitgezählt)."""
        v = np.sin(1.0 + np.arange(self.A.shape[1]))
        nv = np.linalg.norm(v)
        if nv == 0:
            return 0.0
        v = v / nv
        est = 0.0
        for _ in range(iters):
            w = self.tmul(self.mul(v))
            nw = np.linalg.norm(w)
            if nw == 0:
                return 0.0
            est, v = math.sqrt(nw), w / nw
        return est


def adaptive_cap(k, omega, dx, dy, Adx):
    """Obergrenze der Schrittweite aus PDLP für den Schritt (dx, dy) in Iteration k: (1 - (k+1)^-0.3) ||dz||_omega^2 / (2 |dy^T A dx|), ||dz||_omega^2 = omega ||dx||^2 + ||dy||^2 / omega; unendlich, wenn dy^T A dx = 0."""
    inter = abs(float(dy @ Adx))
    dz2 = omega * float(dx @ dx) + float(dy @ dy) / omega
    return (1.0 - (k + 1.0) ** -0.3) * dz2 / (2.0 * inter) if inter > 1e-300 else math.inf


def next_eta(k, eta, cap):
    """Nächste Schrittweite: höchstens die Obergrenze, höchstens (1 + (k+1)^-0.6) mal die aktuelle."""
    return min(cap, (1.0 + (k + 1.0) ** -0.6) * eta)


def restart_due(kkt_c, kkt_start, kkt_prev, k, last_restart):
    """Neustart-Kriterien (KKT-Fehler des Kandidaten): hinreichend (<= 0.2 des Fehlers am letzten Neustart), notwendig (<= 0.8 und der Fehler ist seit der letzten Auswertung gestiegen), künstlich (seit dem letzten Neustart mindestens 0.36 der bisherigen Iterationen)."""
    return kkt_c <= BETA_SUFFICIENT * kkt_start or (kkt_c <= BETA_NECESSARY * kkt_start and kkt_c > kkt_prev) or k - last_restart >= BETA_ARTIFICIAL * k


def primal_weight_update(omega, dx, dy):
    """Geglättetes Primalgewicht: omega <- exp(0.5 ln(dy / dx) + 0.5 ln omega) aus den Bewegungen seit dem letzten Neustart; ohne Bewegung (dx oder dy fast 0) unverändert."""
    if dx > 1e-10 and dy > 1e-10:
        return math.exp(PW_SMOOTHING * math.log(dy / dx) + (1.0 - PW_SMOOTHING) * math.log(omega))
    return omega


def pdlp_flops(nnz, calls, iterations, m, N, precond):
    """Operationsmodell (Multiply-Add = 2): 2 nnz je Matrix-Vektor-Produkt, je Iteration etwa 16 Vektoroperationen der Länge m + N (Schritte, Mittelung, Prüfung), Vorkonditionierung 2 nnz je Sweep (11 Sweeps)."""
    return int(2 * nnz * calls + 16 * iterations * (m + N) + (22 * nnz if precond else 0))


@dataclass
class PDLPResult:
    status: str                                       # "optimal" | "infeasible" | "unbounded" (Strahl nachgerechnet) | "limit" (Iterationsgrenze, nie als Optimum) | "numerical"
    x: tuple = ()                                     # Strukturvariablen (leer ohne Ergebnis)
    x_full: tuple = ()                                # alle Variablen der Standardform (für die Zählung der Nichtnullen)
    y: tuple = ()
    obj: float = float("nan")
    iterations: int = 0
    matvecs: int = 0                                  # Matrix-Vektor-Produkte insgesamt (Norm, Schritte, verworfene Versuche, Zwischenprodukte)
    rejected: int = 0                                 # verworfene Schrittweiten-Versuche
    forced: int = 0                                   # Iterationen, in denen nach MAX_TRIALS Versuchen die letzte Schrittweite trotz verletzter Bedingung genommen wurde (Erwartung: 0)
    restarts: int = 0
    restart_iters: list = field(default_factory=list)
    check_iters: list = field(default_factory=list)   # Iterationen, an denen ausgewertet wurde
    prim_res: list = field(default_factory=list)      # relative Residuen des Kandidaten (besser von aktuell/Mittelwert) am Original
    dual_res: list = field(default_factory=list)
    gap: list = field(default_factory=list)
    cur_max: list = field(default_factory=list)       # größtes der drei Residuen der aktuellen Iterierten
    eta_path: list = field(default_factory=list)      # Schrittweite je Iteration
    omega_path: list = field(default_factory=list)    # Primalgewicht je Iteration
    points: list = field(default_factory=list)        # Strukturvariablen der aktuellen Iterierten je Iteration (nur mit keep; Start zuerst)
    trace: list = field(default_factory=list)         # (eta, Obergrenze) je akzeptierter Iteration (nur mit keep und adaptive)
    avg_points: list = field(default_factory=list)    # Mittelwert je Iteration (nur mit keep und average)
    nnz_x: int = 0                                    # Komponenten der Standardform über NNZ_TOL (relativ)
    nnz_A: int = 0
    m: int = 0
    N: int = 0
    n_struct: int = 0
    norm_est: float = 0.0
    flops: int = 0
    note: str = ""


def _relative(v, ref):
    return float(np.linalg.norm(v)) / (1.0 + ref)


def pdlp(inst, eps=1e-6, max_iter=MAX_ITER, average=True, restart=True, precond=True, adaptive=True, primal_weight=True, keep=False, std=None):
    """PDLP mit fünf einzeln schaltbaren Bausteinen (Überläufe bei extrem schlechter Skalierung werden als numerischer Abbruch gemeldet, nicht als Warnung ausgegeben)."""
    with np.errstate(all="ignore"):
        return _pdlp(inst, eps, max_iter, average, restart, precond, adaptive, primal_weight, keep, std)


def _pdlp(inst, eps, max_iter, average, restart, precond, adaptive, primal_weight, keep, std):
    M0, b0, c0, n, note = std if std is not None else pdlp_form(inst)
    m, N = M0.shape
    nnz = int(np.count_nonzero(M0))
    res = PDLPResult(status="limit", m=m, N=N, n_struct=n, nnz_A=nnz, note=note)
    if m == 0:
        res.status, res.note = "numerical", "keine Bedingungen"
        return res
    A, r, s = precondition(M0) if precond else (np.array(M0, dtype=float), np.ones(m), np.ones(N))
    b, c = b0 / r, c0 / s
    op = MatOp(A)
    norm = op.norm()
    res.norm_est = norm
    if norm == 0.0:
        res.status, res.note = "numerical", "Matrix ist null"
        return res
    nb0, nc0 = float(np.linalg.norm(b0)), float(np.linalg.norm(c0))
    nb, nc = float(np.linalg.norm(b)), float(np.linalg.norm(c))
    omega = nc / nb if primal_weight and nb > 0 and nc > 0 else 1.0
    amax = float(np.abs(A).max())
    eta = 1.0 / amax if adaptive else 0.9 / norm

    def measures(x, y, Ax, Aty):
        """Relative Residuen am Original (primal, dual, Lücke) und die Größen des skalierten Problems (p, d, g) für den KKT-Fehler."""
        p_vec, d_vec = Ax - b, np.maximum(Aty - c, 0.0)
        pobj, dobj = float(c @ x), float(b @ y)
        g = abs(pobj - dobj)
        rel = (_relative(r * p_vec, nb0), _relative(s * d_vec, nc0), g / (1.0 + abs(pobj) + abs(dobj)))
        return rel, (float(np.linalg.norm(p_vec)), float(np.linalg.norm(d_vec)), g)

    def kkt(scaled, w):
        p, d, g = scaled
        return math.sqrt(w * w * p * p + d * d / (w * w) + g * g)

    x, y = np.zeros(N), np.zeros(m)
    Ax, Aty = np.zeros(m), np.zeros(N)
    x_start, y_start = x.copy(), y.copy()
    sx, sy, sAx, sAty, sw = np.zeros(N), np.zeros(m), np.zeros(m), np.zeros(N), 0.0
    kkt_start = kkt(measures(x, y, Ax, Aty)[1], omega)
    kkt_prev = math.inf
    last_restart = 0

    def unscaled(xv, yv):
        return xv / s, yv / r

    def struct(xv):
        return tuple(float(v) for v in (xv / s)[:n])

    if keep:
        res.points.append(np.array(struct(x)))
        if average:
            res.avg_points.append(np.array(struct(x)))
    best = (x, y, Ax, Aty)
    for k in range(1, max_iter + 1):
        for trial in range(MAX_TRIALS):
            tau, sigma = eta / omega, eta * omega
            xn = np.maximum(0.0, x - tau * (c - Aty))
            Axn = op.mul(xn)
            yn = y + sigma * (b - (2.0 * Axn - Ax))
            if not adaptive:
                eta_next = eta
                break
            cap = adaptive_cap(k, omega, xn - x, yn - y, Axn - Ax)
            eta_next = next_eta(k, eta, cap)
            if eta <= cap or trial == MAX_TRIALS - 1:
                res.forced += eta > cap
                if keep:
                    res.trace.append((eta, cap))
                break
            res.rejected += 1
            eta = REJECT_SHRINK * eta_next
        Atyn = op.tmul(yn)
        x, y, Ax, Aty = xn, yn, Axn, Atyn
        res.eta_path.append(eta), res.omega_path.append(omega)
        if average:
            sx, sy, sAx, sAty, sw = sx + eta * x, sy + eta * y, sAx + eta * Ax, sAty + eta * Aty, sw + eta
        eta = eta_next
        res.iterations = k
        if keep:
            res.points.append(np.array(struct(x)))
            if average:
                res.avg_points.append(np.array(struct(sx / sw)))
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            res.status, res.note = "numerical", "Iterierte nicht mehr endlich"
            break
        if k % CHECK_EVERY:
            continue
        cur_rel, cur_sc = measures(x, y, Ax, Aty)
        cand = (x, y, Ax, Aty)
        cand_rel, cand_sc = cur_rel, cur_sc
        if average:
            xa, ya, Axa, Atya = sx / sw, sy / sw, sAx / sw, sAty / sw
            avg_rel, avg_sc = measures(xa, ya, Axa, Atya)
            if kkt(avg_sc, omega) < kkt(cur_sc, omega):
                cand, cand_rel, cand_sc = (xa, ya, Axa, Atya), avg_rel, avg_sc
        res.check_iters.append(k), res.prim_res.append(cand_rel[0]), res.dual_res.append(cand_rel[1]), res.gap.append(cand_rel[2]), res.cur_max.append(max(cur_rel))
        best = cand
        if max(cand_rel) <= eps:
            res.status = "optimal"
            break
        if k % RAY_EVERY == 0:
            x0, y0 = unscaled(x, y)
            if dual_ray(M0, b0, y0, tol=1e-6):
                res.status, res.note = "infeasible", "Strahl y mit M^T y <= 0 und b^T y > 0 gefunden (Farkas-Zertifikat)"
                break
            if primal_ray(M0, c0, x0, tol=1e-6):
                res.status, res.note = "unbounded", "Strahl x >= 0 mit M x = 0 und c^T x < 0 gefunden (unbeschränkt)"
                break
        if restart:
            kkt_c = kkt(cand_sc, omega)
            if restart_due(kkt_c, kkt_start, kkt_prev, k, last_restart):
                xc, yc, Axc, Atyc = cand
                if primal_weight:
                    omega = primal_weight_update(omega, float(np.linalg.norm(xc - x_start)), float(np.linalg.norm(yc - y_start)))
                x, y, Ax, Aty = xc.copy(), yc.copy(), Axc.copy(), Atyc.copy()
                x_start, y_start = x.copy(), y.copy()
                kkt_start = kkt(measures(x, y, Ax, Aty)[1], omega)
                kkt_prev = kkt_start
                sx, sy, sAx, sAty, sw = np.zeros(N), np.zeros(m), np.zeros(m), np.zeros(N), 0.0
                last_restart = k
                res.restarts += 1
                res.restart_iters.append(k)
                continue
            kkt_prev = kkt_c
    else:
        res.status = "limit"
        if res.prim_res:
            parts = {"primal": res.prim_res[-1], "dual": res.dual_res[-1], "Lücke": res.gap[-1]}
            worst = max(parts, key=parts.get)
            res.note = f"Iterationsgrenze erreicht, kein Optimalitätsanspruch (größtes relatives Residuum: {worst} {parts[worst]:.1e})"
    res.matvecs = op.calls
    res.flops = pdlp_flops(nnz, op.calls, res.iterations, m, N, precond)
    if res.status in ("optimal", "limit", "numerical") and res.iterations > 0 and np.all(np.isfinite(best[0])) and np.all(np.isfinite(best[1])):
        xo, yo = unscaled(best[0], best[1])
        res.x, res.x_full, res.y = tuple(float(v) for v in xo[:n]), tuple(float(v) for v in xo), tuple(float(v) for v in yo)
        res.obj = -float(c0 @ xo)
        top = float(np.max(np.abs(xo))) if len(xo) else 0.0
        res.nnz_x = int(np.sum(xo > NNZ_TOL * max(top, 1e-300)))
    return res
