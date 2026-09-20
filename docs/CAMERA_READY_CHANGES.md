# Corrections needed in the camera-ready paper

The checked-in repository is the corrected **code and data artifact**. The
supplied `Main.pdf` and Overleaf ZIP still report historical QAOA results and
must be revised before paper submission. This document is a handoff for that
revision; it does not claim that the conference has approved substantive changes.

The original code was audited at commit
`07f8552028703f8ead7775c8f4810881322dd084`. The public QAOA cost function
read Qiskit measurement bits in the opposite order from the cost Hamiltonian.
An exhaustive check found disagreement on 238 of 256 basis states. The
corrected implementation now agrees on all 256. Its objective is

`E(x) = Σ_i Q_ii x_i + Σ_(i<j) [(Q_ij + Q_ji)/2] x_i x_j`.

Stored `Q` is symmetric, but each pair is counted once. The earlier prose
`xᵀQx` counts pairs twice and needs correction. The circuit construction and
0.12 pair penalty were preserved. See
[`objective_value`](../phase2_qaoa/qubo_encoder.py),
[`_hamiltonian_cost`](../phase2_qaoa/qaoa_runner.py), and the
[exhaustive tests](../phase2_qaoa/tests/test_objective_consistency.py).

## Values to replace

Corrected [ten-seed results](../outputs/reproduction/qaoa_corrected/qaoa_multiseed.json)
were generated from seeds 42–51, twice per seed. The
[part files](../outputs/reproduction/qaoa_corrected/qaoa_multiseed_parts/)
record source hashes, environment versions, resource counts, and the producing
commit. The previous results remain in
[the historical QAOA JSON](../outputs/qaoa_multiseed.json).

| Claim | Historical paper | Corrected result |
|---|---:|---:|
| QAOA failures, mean ± population SD | 3.5 ± 0.8 | **3.5 ± 0.7** |
| QAOA discovery AUC | 133.9 ± 27.9 | **134.6 ± 35.7** |
| Unique simulator states | 44.4 ± 2.7 | **37.8 ± 5.1** |
| Mean first-failure iteration | 1.9 ± 1.1 | **2.1 ± 1.8** |
| Seed 42 failures / recall | 5/18; 28% | **3/18; 17%** |
| Seed 42 AUC / first-failure iteration / unique states | 172.5 / 4 / 45 | **94.5 / 1 / 38** |
| Search simulator calls per seed | 400 | **400** |
| Optimizer shots per seed | claimed ≥1.8M | **1,792,000 measured** |
| Final sampling shots per seed | omitted | **51,200 measured** |
| Total circuit shots per seed | not established | **1,843,200 measured** |

These changes affect the abstract, the QAOA methods/results, resource table,
seed-42 table, Figure 4, Figure 10, their captions, the limitations, and the
conclusion. Use `ROUND_HALF_UP` to one decimal and population SD, as in the
paper. The corrected mean failure count remains below the saved MC mean of
4.2 ± 1.1. This is a result for one tested configuration, not a general QAOA
limit.

The [exact-objective diagnostic](../outputs/reproduction/qaoa_corrected/qubo_diagnostics.json)
finds seven failures in its **50 lowest-energy states**. At QAOA's corrected
seed-42 unique budget of 38 states, it finds four. At the ten actual unique
budgets, the deterministic ranking finds `[4,4,4,4,4,4,7,4,4,4]` failures
(mean 4.3, population SD 0.9; variation is due to the budgets). This ranking
is a specified comparator, **not a ceiling** on arbitrary optimization or
sampling. The diagnostic treats energies equal to 12 decimal places as ties
and uses stable grid order for those ties.

## Resource and control disclosures

The Qiskit search now makes 400 simulator calls and no full-grid catalog calls.
The earlier public program also evaluated a 256-state catalog during setup;
its 400-call counter covered only the search loop. The QUBO consumes persisted
policy failure labels, so zero *new* labels during each QAOA seed does not make
it label-free. Resource counts in the corrected JSON distinguish optimizer
shots, final sampling shots, search calls, setup calls, and unique states.

The teacher/refiner/student pipeline now reuses its 50 search evaluations for
reporting. A standard full-circuit run records 256 teacher-label evaluations,
50 search evaluations, zero extra reporting calls, and 65,536 sampled circuit
shots. Historical warm-start labels are a separate, shared upstream cost.
The policy reproduction measured 11,520 training rollouts and 500 evaluation
rollouts; do not charge these to every downstream seed independently.

The shuffled and random control numbers had already changed between the
reviewed and camera-ready versions because of an artifact repair:
shuffled **+27.6 ± 11.8 → +26.7 ± 14.4**, random
**+15.1 ± 8.5 → +14.1 ± 3.6**. Both exports remain in
[`outputs/`](../outputs/). Disclose this change in the paper itself; the
README provenance note alone is insufficient. The code corrections in this
revision did not replace those arms again.

A full mechanism-control rerun and a full positive-control rerun reproduced
all 290 current curves. The strongest student-readout oracle beats its own
shuffle by **+6.9 ± 7.5** AUC, with bootstrap CI **[2.8,11.8]** and eight
positive, one negative and one tied seed. The circuit versus its shuffle in
the direct target readout is **−10.1 ± 58.7**, CI **[−44.5,28.6]**. The
latter is not an equivalence test or proof that the circuit contains no
candidate-specific signal. The constant and random substitutions recover
part of the full lift; the zero-score arm worsens it. Narrow the blanket
mechanism claims accordingly.

All 16 tests pass. The corrected-artifact checker verifies every basis-state
energy, all ten seed files, the objective-matched comparator, resource totals,
and 144 paired statistical records. The historical manuscript checkers still
pass historical numbers; they do not certify a revised paper. The 10 current
[figure PDFs](../figures/) were regenerated using corrected data; the older
PDFs are archived in [`outputs/historical_figures`](../outputs/historical_figures/).

The original independent audit and complete 79-item old/new manifest remain
in the local handoff package supplied to the authors. After the LaTeX is
revised, compile and visually inspect the eight-page PDF, check every number
against the corrected JSON, and confirm the exact EasyChair title. This
repository cannot by itself make the unchanged manuscript upload-ready.
