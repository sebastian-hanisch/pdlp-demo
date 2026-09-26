"""Konstanten der Demo PDLP: Regler-Bereiche, Stufen, feste Instanzen für die Auswertung, Presets."""
M_MIN, M_MAX, DEFAULT_M = 2, 40, 12
N_MIN, N_MAX, DEFAULT_N = 2, 40, 12
DENSITIES = (0.05, 0.1, 0.2, 0.5, 1.0)                # Dichte der Zufalls-, Misch- und Plateau-Instanzen
DEFAULT_DENSITY_I = 3
SEED_MAX = 999999
DEFAULT_SEED = 35
EPS_EXPS = (2, 3, 4, 5, 6, 7, 8)                      # Genauigkeit 10^-k (relative Residuen und Lücke)
DEFAULT_EPS_I = 2                                     # 1e-4
CAPS = (1000, 5000, 20000, 100000)                    # Iterationsgrenze
DEFAULT_CAP_I = 2
SCALE_EXPS = (0, 2, 4, 6, 8, 10, 12)                  # Spaltenskalierung 10^k (wie in den Stücken 7 bis 9)
DEFAULT_SCALE_I = 0
BLOCK_KEYS = {"average": "average_toggle", "restart": "restart_toggle", "precond": "precond_toggle", "adaptive": "adaptive_toggle", "primal_weight": "pw_toggle"}
BLOCK_HELP = {
    "average": "Laufender, mit der Schrittweite gewichteter Mittelwert der Iterierten seit dem letzten Neustart; er liefert die O(1/k)-Garantie und wird als Kandidat für den Neustart und die Ausgabe genutzt.",
    "restart": "Neustart auf den besseren von aktueller Iterierter und Mittelwert (KKT-Fehler): hinreichend, wenn er auf 0.2 fällt; notwendig (0.8) ohne Fortschritt; künstlich nach 0.36 der bisherigen Iterationen. Ohne Mittelung bleibt ein Neustart wirkungslos.",
    "precond": "Ruiz-Äquilibrierung (10 Sweeps, Max-Norm) und Pock-Chambolle (1-Norm): Zeilen und Spalten der Matrix werden ähnlich groß, das Verfahren läuft auf dem skalierten Problem.",
    "adaptive": "Adaptive Schrittweite: nach jedem Versuch wird geprüft, ob die Schrittweite zur lokalen Lipschitz-Schätzung passt; sonst wird verkleinert und wiederholt (kostet zusätzliche Produkte). Aus: fester Wert 0.9 / ‖A‖.",
    "primal_weight": "Primalgewicht ω: Verhältnis der Schrittweiten für primal und dual; startet bei ‖c‖ / ‖b‖ und wird bei jedem Neustart an die Bewegungen angepasst. Aus: ω = 1.",
}
STEPS = {1: "1 · Der Weg", 2: "2 · Konvergenz", 3: "3 · Die fünf Bausteine", 4: "4 · Gegen Simplex und Innere Punkte", 5: "5 · Grenzen"}
SWEEP_SEEDS = tuple(range(100000, 100005))
SIZE_SIZES = (8, 16, 32, 64, 100)                     # Zufallsinstanzen m = n
SIZE_SEEDS = SWEEP_SEEDS[:3]
LARGE_SIZES = (200, 300, 500)                         # dünne Instanzen (Dichte 2 %), nur PDLP gegen Simplex
LARGE_DENSITY = 0.02
CUBE_SIZES = tuple(range(2, 15))
EPS_SWEEP_EXPS = (2, 4, 6, 8, 10)
SCALE_SWEEP_EXPS = (0, 2, 4, 6, 8, 10, 12)
SCALE_SWEEP_SIZE = 10                                 # Zufallsinstanzen 10 x 10
DETECT_SEEDS = tuple(range(30))
_BASE = {"kind": "random", "m": 12, "n": 12, "seed": 35, "density": 3, "eps": 2, "cap": 2, "scale": 0, "average": True, "restart": True, "precond": True, "adaptive": True, "primal_weight": True, "step": 2}
_OFF = {"average": False, "restart": False, "precond": False, "adaptive": False, "primal_weight": False}
PRESETS = {
    "Lehrbuch: der Weg der Grundform": {**_BASE, "kind": "textbook", **_OFF, "step": 1, "iter_k": 200},
    "Lehrbuch: PDLP mit Neustarts": {**_BASE, "kind": "textbook", "step": 1, "iter_k": 40},
    "Die Grundform ist langsam": {**_BASE, **_OFF, "step": 2},
    "Vorkonditionierung rettet die Spanne": {**_BASE, "m": 10, "n": 10, "scale": 3, "eps": 4, "step": 2},
    "Ohne Vorkonditionierung hängt es": {**_BASE, "m": 10, "n": 10, "scale": 3, "eps": 4, "precond": False, "step": 2},
    "Die fünf Bausteine im Vergleich": {**_BASE, "m": 20, "n": 20, "step": 3},
    "Jede Stelle kostet wenig": {**_BASE, "m": 20, "n": 20, "step": 5},
    "Plateau: keine Ecke": {**_BASE, "kind": "plateau", "m": 10, "n": 12, "step": 5},
    "Gegen Innere Punkte: dünn und groß": {**_BASE, "m": 40, "n": 40, "density": 0, "step": 4},
    "Klee-Minty-Würfel": {**_BASE, "kind": "klee_minty", "n": 14, "step": 4},
    "Unzulässig: Farkas-Strahl": {**_BASE, "kind": "infeasible", "step": 2},
    "Unbeschränkt: Strahl": {**_BASE, "kind": "unbounded", "step": 2},
}
PRESET_HELP = {
    "Lehrbuch: der Weg der Grundform": "Lehrbuchbeispiel (Optimum 36 bei (2, 6)): die Grundform von PDHG braucht bei ε = 10^-4 340 Iterationen und 760 Matrix-Vektor-Produkte (der Simplex 2 Pivots). Im Bild steht Iteration 200: die Ecke ist früh erreicht, doch 334 von 341 Iterierten liegen knapp außerhalb der zulässigen Menge (PDHG erzwingt nur x ≥ 0), und sie kreisen sich langsam ein.",
    "Lehrbuch: PDLP mit Neustarts": "Mit allen fünf Bausteinen: 72 Iterationen und 249 Produkte statt 340 und 760, dabei 8 Neustarts (blaue Rauten) und 25 abgelehnte Schrittweiten. Die Iterierten laufen anfangs weit aus der zulässigen Menge hinaus (bis etwa (7.7, 16.9)); Neustarts holen sie zurück. Der Mittelwert (lila gepunktet) läuft glatter auf die Ecke zu.",
    "Die Grundform ist langsam": "Zufall 12 × 12 (Seed 35): die Grundform braucht bei ε = 10^-4 12.888 Iterationen (25.856 Produkte), PDLP 152 Iterationen (426 Produkte): 85-mal weniger Iterationen. Der Simplex braucht 3 Pivots.",
    "Vorkonditionierung rettet die Spanne": "Zufall 10 × 10, Spalten über 10^6 gestreut: PDLP mit allen fünf Bausteinen löst bei ε = 10^-4 nach 500 Iterationen (18 Neustarts) und findet das Optimum 285.54 des Simplex.",
    "Ohne Vorkonditionierung hängt es": "Dieselbe Instanz ohne Vorkonditionierung erreicht die Genauigkeit auch nach 20.000 Iterationen nicht; der Zielwert liegt 14 % unter dem Optimum (244.58 statt 285.54). Ein Lauf an der Grenze ist kein Ergebnis und wird nie als Optimum gemeldet.",
    "Die fünf Bausteine im Vergleich": "Zufall 20 × 20: die Grundform erreicht 10^-4 in 20.000 Iterationen nicht, PDLP braucht 356 (diese Instanz). Klick auf 'Ablation der fünf Bausteine berechnen' (Median über fünf Instanzen): die Vorkonditionierung allein 3.228 Iterationen, die adaptive Schrittweite allein 11.104, Mittelung, Neustarts oder Primalgewicht allein nichts; alle fünf 284, PDLP ohne einen Baustein 564 bis 1.736.",
    "Jede Stelle kostet wenig": "Zufall 20 × 20, Median über fünf Instanzen: ε = 10^-2 braucht 72 Iterationen, 10^-4 284, 10^-6 652, 10^-8 948 und 10^-10 1.124. Nach den Neustarts fällt der Fehler linear: jede weitere Stelle kostet unter 200 Iterationen. Klick auf 'Genauigkeit verschärfen'.",
    "Plateau: keine Ecke": "Plateau 10 × 12: die Zielfunktion ist parallel zur Gesamtkapazität, eine ganze Fläche ist optimal. Der Simplex liefert eine Ecke mit 10 Nichtnullen (m), PDLP eine Lösung im Inneren der Fläche: 21 bis 22 von 22 Komponenten sind positiv, auch bei ε = 10^-10. Klick auf 'Genauigkeit verschärfen'.",
    "Gegen Innere Punkte: dünn und groß": "Zufall 40 × 40 mit 5 % Dichte: PDLP braucht 236 Iterationen und 653.404 Operationen im Modell, Mehrotra 7 Iterationen und 2.170.931 (dichte Zerlegung: 3.3-mal mehr), der Simplex 5 Pivots und 33.210 (PDLP 19.7-mal mehr). Klick auf 'Operationen über die Größe berechnen'.",
    "Klee-Minty-Würfel": "Würfel n = 14: der Simplex braucht 16.383 Pivots, Mehrotra 15 Iterationen, PDLP 44; im Operationsmodell 14.253.210 gegen 241.350 (Mehrotra) und 75.978 (PDLP): hier ist PDLP am billigsten.",
    "Unzulässig: Farkas-Strahl": "Nach 192 Iterationen ist ein Strahl y mit Mᵀy ≤ 0 und bᵀy > 0 gefunden und nachgerechnet: die Mindestmenge x1 ≥ 6 widerspricht x1 ≤ 4. Das ist ein Beweis, kein Verdacht.",
    "Unbeschränkt: Strahl": "Nach 704 Iterationen ist ein Strahl x ≥ 0 mit Mx = 0 und cᵀx < 0 gefunden und nachgerechnet: der Zielwert wächst ohne Grenze. Ein Beweis, kein Verdacht.",
}


def eps_label(i):
    return f"10^-{EPS_EXPS[i]}"


def cap_label(i):
    return f"{CAPS[i]:,}".replace(",", ".")


def scale_label(i):
    return "unskaliert" if SCALE_EXPS[i] == 0 else f"über 10^{SCALE_EXPS[i]}"


def density_label(i):
    return f"{DENSITIES[i]:.0%}"
