# PDLP – Lineare Programme ohne Faktorisierung – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-pdlp-demo.streamlit.app/)**
Zehntes Stück der **Lineare-Programmierung-Reihe** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", Kind der [Inneren Punkte](https://github.com/sebastian-hanisch/innere-punkte-demo); greift die Skalierungsbefunde aus [Präsolve und Numerik](https://github.com/sebastian-hanisch/praesolve-demo) auf. Simplex und Innere Punkte zerlegen Matrizen (Basis, Cholesky-Faktor). Bei riesigen dünnen LPs wird das zu teuer: die Faktoren füllen sich auf, und man kann sie nicht auf einer GPU parallelisieren. **PDLP** (Applegate u. a., NeurIPS 2021) rechnet nur mit **Matrix-Vektor-Produkten**: ein primal-duales Hybrid-Gradienten-Verfahren (PDHG, Chambolle & Pock 2011) auf dem Sattelpunkt des LPs, je Iteration zwei Produkte, kein Faktorisieren. Die Grundform konvergiert langsam; erst fünf Bausteine (Vorkonditionierung, Mittelung, Neustarts, adaptive Schrittweite, Primalgewicht) machen daraus das Verfahren, das HiGHS, COPT, Xpress und cuOpt anbieten. Die Demo baut jeden Baustein einzeln ein und aus und misst, was er bringt. Vier Fragen: **(1) Der Weg und die Konvergenz** – wie sieht PDHG aus, wie fallen die Residuen? **(2) Die fünf Bausteine** – was bringt jeder? **(3) Gegen Simplex und Innere Punkte** – wo lohnt es sich? **(4) Grenzen** – Genauigkeit, keine Ecke, Skalierung, Unzulässigkeit.

**Einordnung in die Reihe:** die Reihe hat elf Stücke, dies ist das zehnte (Details in `lp-planung/PLAN.md` des Portfolio-Ordners):

```
Tableau-Simplex (Wurzel)                                                                  [gebaut: tableau-simplex-demo]
 ├─ Pivotregeln & Entartung ─ Simplex im schlimmsten und im typischen Fall (Klee-Minty)   [gebaut: pivotregeln-demo, klee-minty-demo]
 ├─ Revised Simplex ─ Präsolve, Skalierung & Numerik                                     [gebaut: revised-simplex-demo, praesolve-demo]
 ├─ Dualität & Sensitivität ─ Dualer Simplex & Neuoptimierung                            [gebaut: lp-dualitaet-demo, dualer-simplex-demo]
 ├─ Ellipsoid-Methode (Kontrast: polynomial in der Theorie)                              [gebaut: ellipsoid-demo]
 └─ Innere Punkte ─ PDLP (Verfahren erster Ordnung) ─ Crossover & Simplex gegen Innere Punkte gegen PDLP
      [gebaut: innere-punkte-demo]   →   [DIESES STÜCK]   →   [nicht gebaut]
```

Ergebnis in Kürze: **Die Grundform von PDHG ist praktisch unbrauchbar, PDLP mit allen fünf Bausteinen nicht – aber im Operationsmodell der Demo gewinnt es nur gegen die dichten Inneren Punkte, nie gegen den Simplex, und ohne Vorkonditionierung scheitert es an Daten in verschiedenen Einheiten schon bei Spalten über 10²** (der Simplex erst ab 10¹², Stück 9). Auf einer Zufallsinstanz mit 12 × 12 braucht die Grundform bei ε = 10⁻⁴ 12.888 Iterationen, PDLP 152 (85-mal weniger); auf 20 × 20 erreicht die Grundform die Genauigkeit in 20.000 Iterationen nie, PDLP im Median nach 284. Von den fünf Bausteinen trägt jeder: nimmt man einen aus PDLP heraus, steigen die Iterationen auf das Doppelte bis Sechsfache; allein zur Grundform hinzugefügt hilft nur die Vorkonditionierung wirklich (3.228 Iterationen), Mittelung und Neustarts wirken nur **zusammen** (ein Neustart ohne Mittelung ändert die Iteration nicht einmal). Jede weitere Stelle Genauigkeit kostet unter 200 Iterationen (ε = 10⁻² bis 10⁻¹⁰: 72 bis 1.124). Die Lösung ist bei eindeutigem Optimum bis auf die grobe Genauigkeit die **Ecke** des Simplex; hat das LP eine optimale **Fläche**, landet PDLP in deren Innerem (21 bis 22 von 22 Komponenten positiv, statt 10 an einer Ecke): das ist der Anlass für **Crossover** (Stück 12). Auf dem Klee-Minty-Würfel braucht PDLP für alle n zwischen 2 und 14 nur 40 bis 76 Iterationen (Simplex: 2ⁿ − 1 Pivots).

| Frage | Ergebnis (Auslastungsplanung als Standard-LP max c·x; Zufallsinstanzen mit Dichte 50 %, 5 feste Instanzen Seeds 100000–100004, Größe und Dichte je Zeile genannt; Referenz: dichter Tableau-Simplex; vollständig deterministisch, kein Zufall im Verfahren) |
|---|---|
| **Stimmt das Verfahren?** | ✅ PDLP liefert auf über 300 Instanzen (Zufall, Mischung mit ≥ und =, Zentrum, Lehrbuch, entartet, Klee-Minty-Würfel n ≤ 10) den Optimalwert von HiGHS (Abweichung < 5·10⁻⁵ relativ bei ε = 10⁻⁶), Residuen am Original nachgerechnet; mindestens 97 % erreichen die Genauigkeit in 20.000 Iterationen, die übrigen enden als "Iterationsgrenze", nie als Optimum. Das Optimum ist Fixpunkt eines PDHG-Schritts; die Schrittweiten-Bedingung, die Mittelung (gegen das explizite gewichtete Mittel) und die Neustart-Regeln (von Hand) sind einzeln geprüft |
| **Grundform gegen PDLP** | Zufall 12 × 12 (Seed 35), ε = 10⁻⁴: **12.888 gegen 152 Iterationen** (25.856 gegen 426 Matrix-Vektor-Produkte; Simplex 3 Pivots). Lehrbuch (2 Variablen): 340 gegen 72 Iterationen, dabei 8 Neustarts und 25 abgelehnte Schrittweiten |
| **Ablation** | Zufall 20 × 20, ε = 10⁻⁴, Median über 5 Instanzen (Iterationen; Läufe an der Grenze 20.000): Grundform **20.000 (0 von 5 erreichen es)**; einzeln zur Grundform hinzugefügt: Mittelung 20.000 (0 von 5), Neustarts 20.000 (0 von 5), **Vorkonditionierung 3.228 (5 von 5)**, adaptive Schrittweite 11.104 (4 von 5), Primalgewicht 20.000 (2 von 5); **alle fünf 284**; alle fünf ohne Mittelung 960, ohne Neustarts 1.108, ohne Vorkonditionierung 1.736, ohne adaptive Schrittweite 760, ohne Primalgewicht 564. Produkte: alle fünf 714, ohne adaptive Schrittweite 1.600 (die adaptive Regel kostet je Iteration ein paar abgelehnte Versuche, spart aber insgesamt mehr als die Hälfte) |
| **Genauigkeitsleiter** | Zufall 20 × 20, Median über 5 Instanzen, alle fünf Bausteine: ε = 10⁻² / 10⁻⁴ / 10⁻⁶ / 10⁻⁸ / 10⁻¹⁰: **72 / 284 / 652 / 948 / 1.124 Iterationen** (714 Produkte bei 10⁻⁴, 2.496 bei 10⁻¹⁰). Nach den Neustarts fällt der Fehler linear: jede weitere Stelle kostet unter 200 Iterationen |
| **Ist das Ergebnis eine Ecke?** | Nichtnullen der Standardform (Dienste + Schlupf, Komponenten über 10⁻⁶ der größten): Zufall 20 × 20 (eindeutiges Optimum), ε = 10⁻² / 10⁻⁴ / 10⁻⁶ / 10⁻⁸ / 10⁻¹⁰: **27 / 26 / 22 / 20 / 20**, die Simplex-Ecke hat 20 (= m): PDLP nähert sich der Ecke erst bei hoher Genauigkeit. **Plateau** 10 × 12 (Zielfunktion parallel zur Gesamtkapazität, eine ganze Fläche ist optimal): **22 / 22 / 22 / 21 / 21** gegen 10 an der Ecke: das Ergebnis liegt im Inneren der Fläche |
| **Skalierung** | Zufall 10 × 10, Spalten über 10^k gestreut, ε = 10⁻⁶, richtig gelöst (von 5): **PDLP mit allen fünf: 5 / 5 / 5 / 5 / 5 bei k = 0 / 2 / 4 / 6 / 8, 4 bei 10, 3 bei 12** (Iterationen im Median der richtig gelösten: 324 / 552 / 704 / 1.372 / 2.428 / 7.098 / 11.168); **PDLP ohne Vorkonditionierung und Grundform: schon bei k = 2 keine einzige Instanz** (bei k = 0 ohne Vorkonditionierung 5 von 5 nach 1.072 Iterationen, Grundform 1 von 5) |
| **Gegen Innere Punkte und Simplex (Modell)** | Zufall m = n, Dichte 50 %, Median über 3 Instanzen, ε = 10⁻⁴; Operationen im Modell von PDLP im Verhältnis zu Mehrotra (dicht) und zum Simplex (dichtes Tableau): n = 8: 7.70 und 115; n = 16: 2.03 und 36.8; n = 32: 1.61 und 34.9; **n = 64: 0.85 und 19.3 (ab hier braucht PDLP weniger als die dichten Inneren Punkte)**; n = 100: 0.76 und 7.9. PDLP-Iterationen 232 / 216 / 492 / 808 / 1.256, Mehrotra 5 / 7 / 8 / 10 / 10. Bei Dichte 5 % liegt der Kreuzungspunkt gegen die Inneren Punkte bei n = 32 |
| **Große dünne Instanzen** | Dichte 2 %, je eine feste Instanz, ε = 10⁻⁴: n = 200 / 300 / 500: PDLP 3.732 / 1.992 / 4.136 Iterationen, der Simplex nur **12 / 20 / 17 Pivots**; PDLP braucht im Modell das **28.2- / 6.8- / 12.0-Fache** des Simplex (n = 500 unter Linux mit neuerem numpy: 18.1-fach, die Iterationszahl ist plattformabhängig): **kein Kreuzungspunkt** im gemessenen Bereich |
| **Klee-Minty-Würfel** | n = 2…14: PDLP **40 bis 76 Iterationen** (n = 2: 52, n = 14: 44), Mehrotra 4 bis 15, Simplex 2ⁿ − 1 Pivots (16.383 bei n = 14); Modell bei n = 14: PDLP 75.978, Mehrotra 241.350, Simplex 14.253.210 Operationen |
| **Unzulässig und unbeschränkt** | 30 konstruierte unzulässige und 30 konstruierte unbeschränkte Zufallsinstanzen (6 × 6), Grenze 20.000: der nachgerechnete **Strahl wird in 30 von 30 bzw. 30 von 30 Fällen gefunden** (Median 320 bzw. 64 Iterationen); nie ein Optimum |

## Vorab-Hypothesen

| Hypothese (vor der Messung) | Ergebnis |
|---|---|
| Ohne Faktorisierung konvergiert die Grundform sublinear und ist praktisch unbrauchbar | **Bestätigt:** 12.888 Iterationen für ε = 10⁻⁴ auf 12 × 12; auf 20 × 20 nach 20.000 nicht am Ziel; ε = 10⁻⁶ erreicht sie auf keiner der gemessenen Größen ab 8 × 8 innerhalb der Grenze |
| Neustarts und Vorkonditionierung tragen, adaptive Schrittweite und Primalgewicht bringen auf kleinen dichten Instanzen wenig | **Widerlegt:** jeder der fünf Bausteine zählt; in PDLP kostet das Weglassen eines Bausteins das Doppelte (Primalgewicht: 564 statt 284 Iterationen) bis Sechsfache (Vorkonditionierung: 1.736) der Iterationen; die adaptive Schrittweite spart trotz abgelehnter Versuche mehr als die Hälfte der Produkte (1.600 gegen 714) |
| Mittelung und Neustarts helfen jeweils | **Nur zusammen:** allein ändert keiner etwas (Neustarts ohne Mittelung: exakt dieselbe Iteration wie die Grundform, denn der Kandidat ist die aktuelle Iterierte). Zusammen mehr als halbieren sie die Iterationen der Grundform |
| Genauigkeit 10⁻⁴ ist schnell, 10⁻⁸ dagegen sehr langsam | **Nur mild:** 10⁻⁴ braucht 284, 10⁻⁸ 948 Iterationen (dreifach), 10⁻¹⁰ 1.124: nach den Neustarts fällt der Fehler linear |
| Das Ergebnis ist keine Ecke (bei 10⁻⁴ mehr Nichtnullen als m) | **Teils widerlegt:** bei eindeutigem Optimum nähert sich PDLP der Ecke (bei 10⁻⁴ noch 26 statt 20 Nichtnullen, ab 10⁻⁸ genau 20); nur bei einer optimalen **Fläche** bleibt es im Inneren (21 bis 22 von 22 Komponenten positiv statt 10 auch bei 10⁻¹⁰) |
| Vorkonditionierung wird erst ab Spalten über 10⁴ nötig | **Widerlegt:** ohne sie scheitert PDLP schon bei 10² auf allen fünf Instanzen; mit ihr hält es bis 10⁸ (5 von 5) |
| PDLP verliert auf kleinen dichten Instanzen und gewinnt auf großen dünnen | **Teils bestätigt:** gegen die dichten Inneren Punkte gewinnt es ab n = 64 (Dichte 50 %) bzw. 32 (5 %); gegen den Simplex nie, auch nicht bei n = 500 und Dichte 2 % (das Verhältnis fällt bei Dichte 50 % von 115 bei n = 8 auf 7.9 bei n = 100; auf den großen dünnen Instanzen mit Dichte 2 % liegt es bei 6.8 bis 28) |
| Auf dem Klee-Minty-Würfel wachsen die Iterationen mit n | **Widerlegt:** 52 bei n = 2, 44 bei n = 14 (dazwischen 40 bis 76): PDLP ist von der Verschachtelung des Würfels unbeeindruckt (im Gegensatz zum Simplex mit 2ⁿ − 1 Pivots) |
| Unzulässigkeit und Unbeschränktheit zeigen sich nur als Verdacht | **Besser als erwartet:** der nachgerechnete Strahl (Farkas bzw. unbeschränkt) wird auf allen 60 konstruierten Instanzen gefunden; die Demo gibt sonst nur "Iterationsgrenze" aus, nie ein Optimum und keinen Verdacht als Ergebnis |

## Was die Demo zeigt

1. **Fünf Schritte** (Schritt-Slider): **Der Weg** (2 Variablen: zulässiges Vieleck, Iterierte, Mittelwert, Neustarts, Slider über die Iteration) → **Konvergenz** (relative Residuen primal, dual und Lücke über die Iterationen, beide Achsen logarithmisch, Neustarts als Rauten, die Grundform grau gestrichelt; Schrittweite η und Primalgewicht ω) → **Die fünf Bausteine** (🔬 auf Abruf: zwölf Konfigurationen, Median über fünf Instanzen) → **Gegen Simplex und Innere Punkte** (Tabelle für die gewählte Instanz; auf Abruf: Operationen über die Größe mit Kreuzungspunkt, große dünne Instanzen bis n = 500, Klee-Minty-Würfel) → **Grenzen** (auf Abruf: Genauigkeitsleiter mit Nichtnullen gegen m, schlechte Skalierung, Erkennung von Unzulässigkeit und Unbeschränktheit).
2. **Instanzen:** Lehrbuch, Zentrum, entartete Ecke, Zufall, Mischung, **Plateau** (optimale Fläche), Klee-Minty-Würfel, Unzulässig, Unbeschränkt.
3. **Regler:** Größe (m, n bis 40), Dichte (5 bis 100 %), Genauigkeit ε (10⁻² bis 10⁻⁸), Iterationsgrenze (1.000 bis 100.000; ein Lauf an der Grenze ist kein Ergebnis), Spalten-Skalierung (bis 10¹²), fünf **Baustein-Schalter** (Mittelung, Neustarts, Vorkonditionierung, adaptive Schrittweite, Primalgewicht).
4. **Ergebnis:** jedes Ergebnis wird gegen den Simplex geprüft (Optimalwert); Unzulässigkeit und Unbeschränktheit nur mit nachgerechnetem Strahl.

Presets (12): Lehrbuch: der Weg der Grundform, Lehrbuch: PDLP mit Neustarts, Die Grundform ist langsam, Vorkonditionierung rettet die Spanne, Ohne Vorkonditionierung hängt es, Die fünf Bausteine im Vergleich, Jede Stelle kostet wenig, Plateau: keine Ecke, Gegen Innere Punkte: dünn und groß, Klee-Minty-Würfel, Unzulässig: Farkas-Strahl, Unbeschränkt: Strahl.

## Messwerte der Presets

| Preset | Einstellungen | Ergebnis |
|---|---|---|
| **Lehrbuch: der Weg der Grundform** | 2 Variablen, alle Bausteine aus, ε = 10⁻⁴ | 340 Iterationen, 760 Produkte; 334 von 341 Iterierten außerhalb der zulässigen Menge (PDHG erzwingt nur x ≥ 0); Simplex 2 Pivots |
| **Lehrbuch: PDLP mit Neustarts** | alle fünf | 72 Iterationen, 249 Produkte, 8 Neustarts, 25 abgelehnte Schrittweiten; die Iterierten laufen anfangs bis etwa (7.7, 16.9) hinaus |
| **Die Grundform ist langsam** | Zufall 12 × 12, Grundform | 12.888 gegen 152 Iterationen, 25.856 gegen 426 Produkte |
| **Vorkonditionierung rettet die Spanne** | 10 × 10, Spalten über 10⁶ | 500 Iterationen, 18 Neustarts, Optimum 285.54 |
| **Ohne Vorkonditionierung hängt es** | dieselbe Instanz, Vorkonditionierung aus | Iterationsgrenze (20.000), Zielwert 244.58 statt 285.54 (14 % daneben), kein Ergebnis |
| **Die fünf Bausteine im Vergleich** | Zufall 20 × 20 | diese Instanz 356 Iterationen (Grundform: Grenze); Ablation: siehe oben |
| **Jede Stelle kostet wenig** | Zufall 20 × 20 | 72 / 284 / 652 / 948 / 1.124 Iterationen für ε = 10⁻² … 10⁻¹⁰ |
| **Plateau: keine Ecke** | Plateau 10 × 12 | 21 bis 22 von 22 Komponenten positiv, Ecke: 10 |
| **Gegen Innere Punkte: dünn und groß** | Zufall 40 × 40, Dichte 5 % | PDLP 236 Iterationen / 653.404 Operationen; Mehrotra 7 / 2.170.931; Simplex 5 Pivots / 33.210 |
| **Klee-Minty-Würfel** | n = 14 | PDLP 44, Mehrotra 15 Iterationen, Simplex 16.383 Pivots |
| **Unzulässig** | Fixture | Strahl nach 192 Iterationen |
| **Unbeschränkt** | Fixture | Strahl nach 704 Iterationen |

## Modell und Verfahren

- **Instanz** (`pdl_scenario.py`): die Auslastungsplanung der Vorgängerstücke (wortgleich aus `innere-punkte-demo` kopiert) plus **Plateau**: Zufallsinstanz mit c = 2·(erste, dichte Zeile von A): Zielfunktion und Gesamtkapazität sind parallel, jede zulässige Lösung mit A₀x = b₀ ist optimal.
- **PDLP** (`pdl_pdlp.py`): Standardform min c̃ᵀx unter M x = b, x ≥ 0 (Schlupf/Überschuss angehängt, keine Rangprüfung: PDHG braucht unabhängige Zeilen nicht). Iteration x⁺ = max(0, x − τ(c̃ − Mᵀy)), y⁺ = y + σ(b − M(2x⁺ − x)) mit τ = η/ω, σ = ηω; Grundform η = 0.9 / ‖M‖₂ (Potenzmethode, fest gestartet). **Vorkonditionierung:** Ruiz (10 Sweeps, Max-Norm), dann Pock-Chambolle (1-Norm). **Mittelung:** mit der Schrittweite gewichteter Mittelwert seit dem letzten Neustart (Produkte per Linearität mitgeführt: kein zusätzliches Produkt). **Neustarts:** alle vier Iterationen wird der KKT-Fehler (√(ω²‖Mx − b‖² + ‖(Mᵀy − c)⁺‖²/ω² + Lücke²)) des besseren von Iterierter und Mittelwert mit dem am letzten Neustart verglichen: hinreichend (0.2), notwendig ohne Fortschritt (0.8), künstlich nach 0.36 der bisherigen Iterationen. **Adaptive Schrittweite** und **Primalgewicht** wie in Applegate u. a. (Formeln im Programmtext und in der App). **Abbruch:** relative Residuen und Lücke ≤ ε, am Original aus den Zwischenprodukten des skalierten Problems berechnet (M x − b = r∘(M′x′ − b′) usw.: keine zusätzlichen Produkte). **Strahltests** (Farkas, unbeschränkt) alle 64 Iterationen mit den äquilibrierten Zertifikaten aus Stück 8.
- **Auswertung** (`pdl_evaluation.py`): Ablation (zwölf Konfigurationen), Genauigkeitsleiter mit Nichtnullen, Größen-Sweep im Operationsmodell (2·nnz je Produkt und etwa 16 Vektoroperationen je Iteration; Innere Punkte und Simplex dicht wie in Stück 8), große dünne Instanzen, Skalierung, Würfel, Erkennung.

## Was nicht funktioniert hat / Grenzen

- **Die reine Schrittweiten-Regel lehnt zu viel ab.** Mit der Regel aus dem Paper (auf die Obergrenze verkleinern und neu versuchen) verkleinert sich die Schrittweite geometrisch gegen einen Fixpunkt und wird bis zur Rundungsgenauigkeit immer wieder knapp abgelehnt: auf sechs Instanzen (ε = 10⁻⁶) 14.589 abgelehnte Versuche und 23.988 Produkte bei 4.460 Iterationen. Die Demo versucht nach einer Ablehnung nur 0.9 der Obergrenze (**Abweichung vom Paper**): 778 abgelehnte Versuche, 11.602 Produkte, 5.172 Iterationen: die Produkte halbieren sich, die Iterationen steigen um 16 %.
- **Nur die Grundform von PDLP.** Kein Feasibility Polishing, keine GPU, keine Halpern-Varianten (cuPDLPx nur genannt), kein Präsolve; die Auswertung von Abbruch und Neustart alle vier Iterationen statt alle 64 (die Instanzen sind klein).
- **Dichte Speicherung trotz "dünn".** Die Matrix liegt dicht im Speicher; gezählt werden nur die Nichtnullen (2·nnz je Produkt). Das Operationsmodell ist kein Wandzeit-Vergleich; **Innere Punkte und Simplex sind dicht modelliert**, echte Löser faktorisieren dünn: der Kreuzungspunkt gegen sie überschätzt den Vorteil von PDLP.
- **Synthetische, gutartige Instanzen.** Der Simplex braucht auf diesen Zufallsinstanzen wenige Pivots (3 bei 12 × 12, 12 bis 20 bei n = 200 bis 500 mit Dichte 2 %, 108 im Median bei n = 100 mit Dichte 50 %): er ist hier schwer zu schlagen; Instanzen mit vielen Pivots und riesigen dünnen Matrizen (Netlib, MIPLIB) hat die Demo nicht.
- **Der Kreuzungspunkt gegen den Simplex ist nicht gemessen, nur nicht gefunden:** bis n = 500 (Dichte 2 %) braucht PDLP im Modell 6.8- bis 28-mal so viel; darüber sagt die Demo nichts.
- **Keine Ecke.** Basis, Duale mit Ranging und Warmstart gibt es erst mit Crossover ([crossover-demo](https://github.com/sebastian-hanisch/crossover-demo), Stück 11).
- **Plattformabhängigkeit.** Iterationszahlen, Neustart-Zahlen und Nichtnull-Zahlen können unter Windows und Linux um einzelne Prozent abweichen; die Tests prüfen dort Bänder. Ganzzahl-Logik (Zahl der Pivots des Würfels, Kreuzungspunkt in Stufen) ist exakt.
- **Erste Fassung meldete "Verdacht".** Bei Erreichen der Iterationsgrenze hatte ich zunächst einen Status "suspect" vergeben, wenn ein Residuum groß blieb; das traf aber auch die langsame Grundform auf zulässigen LPs und wäre irreführend gewesen. Jetzt: "Iterationsgrenze" ohne Optimalitätsanspruch, dazu das größte Residuum als Hinweis.

## Verifikation

- `tests/test_algorithm.py`: **PDLP gegen HiGHS auf über 300 Instanzen**; **Fixpunkt** (das Optimum bleibt bei einem PDHG-Schritt unverändert, für drei Kombinationen von η und ω) und ein Schritt von Hand auf dem Lehrbuchbeispiel; Norm-Schätzung gegen `numpy` (5 %); Grundform-Schrittweite erfüllt τσ‖M‖² ≤ 1, adaptive Regel erfüllt in **jeder** Iteration ihre Bedingung (Formel von Hand geprüft); **laufender Mittelwert gegen das explizite gewichtete Mittel**; Neustart-Regeln von Hand (hinreichend, notwendig, künstlich); Vorkonditionierung **exakt rückrechenbar** (10⁻¹²), Nullzeilen/-spalten; Primalgewicht-Formel; **Buchführung der Produkte** gegen eine unabhängige Formel (2·40 + 2·Iterationen + abgelehnte Versuche); Iterationsgrenze exakt; Determinismus; konstruierte unzulässige und unbeschränkte Instanzen enden nie als Optimum; Sonderfälle (m = 1, n = 1, c = 0, doppelte und Nullzeilen); Kopien treu (Simplex Zentrum 720, Mehrotra 5 bis 9 Iterationen).
- `tests/test_scenario.py`, `test_evaluation.py`, `test_presets.py` (jede Zahl der Hilfetexte), `test_claims.py` (jede Zahl aus README und App über die echten `ev.*`-Funktionen; Iterationszahlen nur als Bänder), `test_app.py` (Streamlit-AppTest: Voreinstellung, jedes Preset, jeder Schritt für jede Instanz, jeder Baustein-Schalter, Iterations-Slider, Regler-Randwerte, Permalink-Grenzen, bedingte Regler, Berechnungen auf Abruf, Footer).
- Für die Prüfung genügt **pytest**; `scipy` dient nur als Gegenprobe (`requirements-dev.txt`), die App braucht nur numpy, pandas, plotly und streamlit.

## Lokal starten

```bash
python -m venv venv && venv/Scripts/activate  # Windows; Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/ -W error::SyntaxWarning`.

## Literatur

- Applegate, D., Díaz, M., Hinder, O., Lu, H., Lubin, M., O'Donoghue, B., & Schudy, W. (2021). *Practical large-scale linear programming using primal-dual hybrid gradient.* Advances in Neural Information Processing Systems 34.
- Chambolle, A., & Pock, T. (2011). *A first-order primal-dual algorithm for convex problems with applications to imaging.* Journal of Mathematical Imaging and Vision 40(1), 120–145.
- Lu, H., & Yang, J. (2023). *cuPDLP.jl: a GPU implementation of restarted primal-dual hybrid gradient for linear programming in Julia.* arXiv 2311.12180 (nur genannt).

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
