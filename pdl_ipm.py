"""Innere-Punkte-Verfahren (primal-dual, Start ohne Zulässigkeit) für  max c·x,  A x (<=|>=|=) b,  x >= 0.

Standardform: min c~^T z  unter  M z = b~,  z >= 0  (Schlupf- und Überschussvariablen angehängt, c~ = -c). Je Iteration ein Newton-Schritt auf
    M dz = r_p,   M^T dy + ds = r_d,   S dz + Z ds = sigma mu e - Z S e - (Korrektor)
(Z = diag(z), S = diag(s), r_p = b~ - M z, r_d = c~ - M^T y - s), gelöst über die Normalgleichungen  M (S^-1 Z) M^T dy = rhs  per Cholesky.
Verfahren: "affine" (sigma = 0: primal-duales Affine Scaling), "short" (fester Faktor sigma = 1 - 0.4/sqrt(N), Schrittlänge 0.9 der Maximallänge), "long" (sigma = 0.1, Schrittlänge bis in die
Umgebung N_-inf(gamma) zurückgenommen), "mehrotra" (Prädiktor-Korrektor). Alle bleiben strikt im Inneren (z, s > 0)."""

import math
from dataclasses import dataclass, field

import numpy as np

import pdl_scenario as S

LE, GE, EQ = S.LE, S.GE, S.EQ
METHODS = ("affine", "short", "long", "mehrotra")
GAMMA = 1e-3                                # Umgebung N_-inf(gamma) des Langschritt-Verfahrens: z_j s_j >= gamma mu
STEP_FRACTION = 0.9                         # Anteil der Maximallänge bis zum Rand (short, affine, long)
DIVERGENCE = 1e9                            # Iterate über diesem Vielfachen der Daten: Verdacht auf Unzulässigkeit oder Unbeschränktheit
MAX_ITER = 1000


def _equilibrate(M, sweeps=3):
    """Zeilen- und Spaltenskalen (Normen), so dass M / (r c^T) Einträge der Größe 1 hat; nur für Rangtest und Zertifikate (die Iteration selbst rechnet unskaliert)."""
    r, c = np.ones(M.shape[0]), np.ones(M.shape[1])
    for _ in range(sweeps):
        r = np.linalg.norm(M / c, axis=1)
        r[r == 0] = 1.0
        c = np.linalg.norm(M / r[:, None], axis=0)
        c[c == 0] = 1.0
    return r, c


def standard_form(inst):
    """Standardform: Matrix M (m x N), rechte Seite b, Kosten c (Minimierung), Zahl der Strukturvariablen n; Schlupf je <=-Zeile (+1), Überschuss je >=-Zeile (-1). Abhängige Zeilen werden entfernt, wenn ihre
    rechte Seite verträglich ist (Rückgabe der Notiz), sonst ist die Instanz unzulässig (Rückgabe None für M)."""
    A, b, c = inst.arrays()
    m, n = A.shape
    extra = [i for i, s in enumerate(inst.senses) if s != EQ]
    M = np.hstack([A, np.zeros((m, len(extra)))])
    for k, i in enumerate(extra):
        M[i, n + k] = 1.0 if inst.senses[i] == LE else -1.0
    cost = np.concatenate([-c, np.zeros(len(extra))])
    rn, cn = _equilibrate(M)
    Mn, bn = M / rn[:, None] / cn, b / rn                                                     # Zeilen und Spalten normiert: Rangtest und Verträglichkeit unabhängig von der Skalierung
    keep, note = [], ""
    basis = np.zeros((0, M.shape[1]))
    for i in range(m):
        cand = np.vstack([basis, Mn[i]])
        if np.linalg.matrix_rank(cand, tol=1e-9) > len(keep):
            keep.append(i)
            basis = cand
    if len(keep) < m:
        drop = [i for i in range(m) if i not in keep]
        coef = np.linalg.lstsq(Mn[keep].T, Mn[drop].T, rcond=None)[0]                          # abhängige Zeilen als Linearkombination
        if not np.allclose(coef.T @ bn[keep], bn[drop], atol=1e-7 * (1 + np.abs(bn).max())):
            return None, b, cost, n, f"Zeilen {drop} widersprechen den übrigen (unzulässig)"
        note = f"{len(drop)} abhängige Zeile(n) entfernt"
        M, b = M[keep], b[keep]
    return M, b, cost, n, note


def flops_per_iteration(m, N, solves=1):
    """Operationsmodell je Iteration (Multiply-Add = 2): Normalmatrix M D M^T (2 m^2 N), Cholesky (m^3 / 3), je rechte Seite zwei Dreieckslösungen (2 m^2), Matrix-Vektor-Produkte und Residuen (8 m N), Vektoren (10 N)."""
    return int(2 * m * m * N + m ** 3 / 3.0 + solves * 2 * m * m + 8 * m * N + 10 * N)


@dataclass
class IPMResult:
    status: str                                       # "optimal" | "infeasible" | "unbounded" (Strahl nachgerechnet) | "suspect" | "numerical" | "limit"
    x: tuple = ()                                     # Strukturvariablen (leer ohne Ergebnis)
    obj: float = float("nan")                         # Zielwert der Maximierung
    y: tuple = ()
    iterations: int = 0
    mu_path: list = field(default_factory=list)       # mu je Iteration (vor dem Schritt)
    gap_path: list = field(default_factory=list)      # relative Lücke |c z - b y| / (1 + |b y|)
    prim_res: list = field(default_factory=list)      # ||r_p|| / (1 + ||b||)
    dual_res: list = field(default_factory=list)      # ||r_d|| / (1 + ||c||)
    step_p: list = field(default_factory=list)
    step_d: list = field(default_factory=list)
    sigma: list = field(default_factory=list)
    points: list = field(default_factory=list)        # Strukturvariablen je Iterierter (nur mit keep; Start zuerst)
    cond: list = field(default_factory=list)          # Kondition der Normalmatrix je Iteration (nur mit keep_cond)
    min_comp: list = field(default_factory=list)      # min z_j s_j / mu je Iteration (Zentrierung)
    flops: int = 0
    n_struct: int = 0
    m: int = 0
    N: int = 0
    note: str = ""


def _max_step(v, dv):
    neg = dv < 0
    return float(np.min(-v[neg] / dv[neg])) if np.any(neg) else math.inf


def dual_ray(M, b, y, tol=1e-7):
    """Unzulässigkeits-Zertifikat: ein y mit M^T y <= 0 und b^T y > 0 (Farkas), geprüft am normierten Iterierten in zeilen- und spaltenskalierten Größen (unabhängig von der Skalierung der Daten)."""
    r, c = _equilibrate(M)
    yt = y * r
    n = np.linalg.norm(yt)
    if n == 0:
        return False
    yh = yt / n
    Mt = M / r[:, None] / c
    return bool(float((b / r) @ yh) > tol * max(np.linalg.norm(b / r), 1e-300) and np.max(Mt.T @ yh) <= tol)


def primal_ray(M, c, z, tol=1e-7):
    """Unbeschränktheits-Zertifikat: ein z >= 0 mit M z = 0 und c^T z < 0, geprüft am normierten Iterierten in skalierten Größen."""
    r, cs = _equilibrate(M)
    zt = z * cs
    n = np.linalg.norm(zt)
    if n == 0:
        return False
    zh = zt / n
    Mt = M / r[:, None] / cs
    ct = c / cs
    return bool(np.linalg.norm(Mt @ zh) <= tol and float(ct @ zh) < -tol * max(np.linalg.norm(ct), 1e-300))


def newton_direction(M, z, s, L, r_p, r_d, r_c):
    """Newton-Richtung des KKT-Systems  M dz = r_p,  M^T dy + ds = r_d,  S dz + Z ds = r_c  über die Normalgleichungen  M (S^-1 Z) M^T dy = r_p + M (S^-1 Z r_d - S^-1 r_c)  (L: Cholesky-Faktor)."""
    d = z / s
    rhs = r_p + M @ (d * r_d - r_c / s)
    dy = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
    ds = r_d - M.T @ dy
    dz = (r_c - z * ds) / s
    return dz, dy, ds


def mehrotra_start(M, b, c):
    """Startpunkt nach Mehrotra: kleinste-Quadrate-Lösung, dann auf positive Werte verschoben und zentriert."""
    MMt = M @ M.T
    try:
        z = M.T @ np.linalg.solve(MMt, b)
        y = np.linalg.solve(MMt, M @ c)
    except np.linalg.LinAlgError:                                                            # numerisch singulär (extrem schlechte Skalierung): Kleinste-Quadrate-Lösung
        z = np.linalg.lstsq(M, b, rcond=None)[0]
        y = np.linalg.lstsq(M.T, c, rcond=None)[0]
    s = c - M.T @ y
    dz = max(-1.5 * z.min(), 0.0)
    ds = max(-1.5 * s.min(), 0.0)
    zh, sh = z + dz, s + ds
    dzh = 0.5 * float(zh @ sh) / max(float(sh.sum()), 1e-300)
    dsh = 0.5 * float(zh @ sh) / max(float(zh.sum()), 1e-300)
    return zh + dzh, y, sh + dsh


def ipm(inst, method="mehrotra", eps=1e-8, max_iter=MAX_ITER, keep=False, keep_cond=False, start=None, start_factor=1.0, std=None):
    """Innere-Punkte-Verfahren (Überläufe bei extrem schlecht skalierten Instanzen werden als numerischer Abbruch gemeldet, nicht als Warnung ausgegeben)."""
    with np.errstate(all="ignore"):
        return _ipm(inst, method, eps, max_iter, keep, keep_cond, start, start_factor, std)


def _ipm(inst, method, eps, max_iter, keep, keep_cond, start, start_factor, std):
    """Rumpf von `ipm`. eps: relative Genauigkeit von Lücke und Residuen. start: (z, y, s) für Tests; start_factor vergrößert den Mehrotra-Startpunkt (z und s)."""
    if method not in METHODS:
        raise ValueError(method)
    M, b, c, n, note = std if std is not None else standard_form(inst)
    if M is None:
        return IPMResult(status="infeasible", n_struct=n, note=note)
    m, N = M.shape
    res = IPMResult(status="limit", n_struct=n, m=m, N=N, note=note)
    if m == 0:
        return IPMResult(status="numerical", n_struct=n, note="keine Bedingungen")
    if start is None:
        z, y, s = mehrotra_start(M, b, c)
        z, s = z * start_factor, s * start_factor
    else:
        z, y, s = (np.array(v, dtype=float) for v in start)
    nb, nc = 1.0 + np.linalg.norm(b), 1.0 + np.linalg.norm(c)
    sigma_short = 1.0 - 0.4 / math.sqrt(N)
    if keep:
        res.points.append(z[:n].copy())
    for it in range(1, max_iter + 1):
        mu = float(z @ s) / N
        r_p, r_d = b - M @ z, c - M.T @ y - s
        pobj, dobj = float(c @ z), float(b @ y)
        gap = abs(pobj - dobj) / (1.0 + abs(dobj))
        pr, dr = np.linalg.norm(r_p) / nb, np.linalg.norm(r_d) / nc
        res.mu_path.append(mu), res.gap_path.append(gap), res.prim_res.append(pr), res.dual_res.append(dr)
        res.min_comp.append(float(np.min(z * s)) / mu if mu > 0 else 0.0)
        res.iterations = it - 1
        if gap <= eps and pr <= eps and dr <= eps:
            res.status = "optimal"
            break
        if it >= 3 and dual_ray(M, b, y):
            res.status, res.note = "infeasible", "Strahl y mit M^T y <= 0 und b^T y > 0 gefunden (Farkas-Zertifikat)"
            break
        if it >= 3 and primal_ray(M, c, z):
            res.status, res.note = "unbounded", "Strahl z >= 0 mit M z = 0 und c^T z < 0 gefunden (unbeschränkt)"
            break
        if np.max(np.abs(z)) > DIVERGENCE * nb or np.max(np.abs(s)) > DIVERGENCE * nc:
            res.status, res.note = "suspect", "Iterierte laufen davon (Verdacht auf Unbeschränktheit, kein Beweis)"
            break
        if np.max(np.abs(y)) > DIVERGENCE * nc:
            res.status, res.note = "suspect", "Duale laufen davon (Verdacht auf Unzulässigkeit, kein Beweis)"
            break
        if len(res.step_p) >= 8 and max(max(res.step_p[-8:]), max(res.step_d[-8:])) < 1e-6:
            res.status, res.note = "suspect", "Stillstand: die Schritte sind winzig, ein Residuum fällt nicht (Verdacht auf Unzulässigkeit oder Unbeschränktheit, kein Beweis)"
            break
        d = z / s
        try:
            Mat = (M * d) @ M.T
            Mat = 0.5 * (Mat + Mat.T)
            if keep_cond:
                res.cond.append(float(np.linalg.cond(Mat)))
            L = np.linalg.cholesky(Mat)
        except np.linalg.LinAlgError:
            try:
                Mat = Mat + 1e-12 * np.trace(Mat) / m * np.eye(m)
                L = np.linalg.cholesky(Mat)
                res.note = "Normalmatrix regularisiert"
            except np.linalg.LinAlgError:
                res.status, res.note = "numerical", "Normalmatrix nicht mehr positiv definit"
                break

        def newton(r_c):
            return newton_direction(M, z, s, L, r_p, r_d, r_c)

        solves = 1
        if method == "mehrotra":
            dz_a, dy_a, ds_a = newton(-z * s)
            ap, ad = min(1.0, _max_step(z, dz_a)), min(1.0, _max_step(s, ds_a))
            mu_aff = float((z + ap * dz_a) @ (s + ad * ds_a)) / N
            sigma = (mu_aff / mu) ** 3
            dz, dy, ds = newton(sigma * mu - z * s - dz_a * ds_a)
            solves = 2
        else:
            sigma = {"affine": 0.0, "short": sigma_short, "long": 0.1}[method]
            dz, dy, ds = newton(sigma * mu - z * s)
        ap_max, ad_max = _max_step(z, dz), _max_step(s, ds)
        frac = 0.9995 if method == "mehrotra" else STEP_FRACTION
        ap, ad = min(1.0, frac * ap_max), min(1.0, frac * ad_max)
        if method == "long":
            a = min(ap, ad)
            while a > 1e-10:
                zn, sn = z + a * dz, s + a * ds
                if np.all(zn * sn >= GAMMA * float(zn @ sn) / N):
                    break
                a *= 0.5
            ap = ad = a
        z, y, s = z + ap * dz, y + ad * dy, s + ad * ds
        res.step_p.append(ap), res.step_d.append(ad), res.sigma.append(sigma)
        res.flops += flops_per_iteration(m, N, solves)
        if keep:
            res.points.append(z[:n].copy())
        if not (np.all(np.isfinite(z)) and np.all(np.isfinite(s)) and np.all(z > 0) and np.all(s > 0)):
            res.status, res.note = "numerical", "Iterierte nicht mehr strikt positiv"
            break
        res.iterations = it
    else:
        res.status = "limit"
        if pr > 1e-6 or dr > 1e-6:                                                        # kein Ergebnis in Sicht: die Residuen fallen nicht
            res.status, res.note = "suspect", "Iterationsgrenze, ein Residuum fällt nicht (Verdacht auf Unzulässigkeit oder Unbeschränktheit, kein Beweis)"
    if res.status in ("optimal", "limit", "numerical") and res.iterations > 0 and np.all(np.isfinite(z)) and np.all(np.isfinite(y)):
        res.x, res.y = tuple(float(v) for v in z[:n]), tuple(float(v) for v in y)
        res.obj = -float(c @ z)
    return res


def central_point(M, b, c, mu, start=None, iterations=40):
    """Zentralpfad-Punkt zu mu: das Newton-Verfahren auf  M z = b,  M^T y + s = c,  Z S e = mu e  (Zentrierung mit sigma = 1 und festem Ziel mu), von einem inneren Startpunkt aus (Standard: der Mehrotra-Start)."""
    z, y, s = mehrotra_start(M, b, c) if start is None else (np.array(v, dtype=float) for v in start)
    for _ in range(iterations):
        r_p, r_d, r_c = b - M @ z, c - M.T @ y - s, mu - z * s
        if max(np.linalg.norm(r_p), np.linalg.norm(r_d), np.linalg.norm(r_c) / max(mu, 1e-300)) <= 1e-10 * (1 + np.linalg.norm(b)):
            break
        Mat = (M * (z / s)) @ M.T
        L = np.linalg.cholesky(0.5 * (Mat + Mat.T))
        dz, dy, ds = newton_direction(M, z, s, L, r_p, r_d, r_c)
        a = min(1.0, 0.9 * min(_max_step(z, dz), _max_step(s, ds)))
        z, y, s = z + a * dz, y + a * dy, s + a * ds
    return z, y, s
