# QTD-ICTAI: reproduction artifact

Code, scripts, figure sources, and persisted per-seed outputs for

> S. B. Malluvajhula and M. Farooque, "Do Hybrid Quantum Components Improve Rare-Event
> Failure Ranking? A Controlled Evaluation with Mechanism and Budget Controls,"
> IEEE ICTAI 2026.

All circuits are noiseless statevector simulations (Qiskit `StatevectorSampler`); no
quantum hardware is needed. Everything runs on a laptop CPU.

## Setup (Linux / macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # or: pip install -r requirements.lock  (exact versions)
```

Windows (PowerShell): `python -m venv .venv; .venv\Scripts\Activate.ps1; pip install -r requirements.txt`

Tested with Python 3.11, qiskit 1.4.6, qiskit-aer 0.17.2, qiskit-algorithms 0.4.0, numpy 2.x.
The scripts record `git rev-parse HEAD` in their outputs, so run them inside a git clone.

## Quick check (about 2 minutes)

```bash
python -m pytest -q                              # unit tests
python run_traceability_checks.py                # numbers carried over from the reviewed paper vs outputs/*.json
python analysis/check_camera_ready_numbers.py    # numbers added or changed in the camera-ready
```

Both checkers must end with `ALL PASS`.

## Reproduce the results

| Paper item | Command | Output (under `outputs/`) | Time |
| --- | --- | --- | --- |
| Policy weights and rollout failures (Sec. III-A) | `python phase1/train.py && python phase1/evaluate.py` | `phase1/policy_weights.npz`, `failure_states.json` | seconds |
| MC baselines (Tables II-III, Sec. VI-C) | `python run_mc_no_replacement_analysis.py` | `mc_no_replacement.json` | seconds |
| QAOA, one seed at a time (Sec. VI-A) | `python run_qaoa_multiseed.py --seed 42` ... `--seed 51`, then `--assemble` | `qaoa_multiseed_parts/`, `qaoa_multiseed.json` | about 12 min per seed |
| QUBO diagnostics and exact minimizer (Secs. III-B, VI-A) | `python analysis/qubo_diagnostics.py` | `qubo_diagnostics.json` | seconds |
| Mechanism controls (Sec. VI-B1) | `python run_mechanism_controls.py --output-name mechanism_controls_rerun.json` | as named | about 2 min |
| Positive control, constant and zero-score arms (Secs. VI-B1, VI-B2) | `python run_positive_control.py --output-name positive_control_rerun.json` | as named | about 2 min |
| Fixed-ego vs. policy failures, warm-start overlap (Sec. VI-B3) | `python analysis/policy_failure_overlap.py` | `policy_failure_overlap.json` | seconds |
| Label-disjoint splits (Sec. VI-C) | `python run_split_experiment.py --shots 1024 --output-name split_results_1024_rerun.json` | as named | about 1 min |
| CEM baseline (Sec. VI-C1) | `python run_adaptive_baseline.py` | `adaptive_baseline*.json` | seconds |
| Cold start (Sec. VI-C2) | `python run_coldstart_qtd.py` | `coldstart_qtd.json` | minutes |
| 8-qubit sweep, QTDv2, QTDv3 (Sec. VI-D, exploratory) | `python run_8dim_quantum_sweep.py`, `python run_qtd_v2_matched_benchmark.py`, `python run_qtd_v3_stochastic_suite.py` | `quantum_8dim_sweep.json`, `reproduction/` | minutes |
| Derived statistics cited in the text | `python analysis/camera_ready_stats.py` | `camera_ready_stats.json` | seconds |
| All figures | `python figures/make_paper_figures.py` | `figures/*.pdf` | seconds |

Scripts refuse to overwrite a canonical output unless `--force` is passed; use `--output-name`
(where available) to write a rerun next to the persisted file and compare.

## Provenance notes

* **Repair of the first export.** The first public export of this repository was truncated by the
  export tool: `phase3_quantum_tree/quantum_layer.py` (inside `_qiskit_refine`),
  `run_split_experiment.py` (end of `main`), trailing NUL bytes in `run_traceability_checks.py`,
  and two JSON files. The missing code was reconstructed from the in-file specification and
  validated against the persisted outputs; see `outputs/repair_validation.json`. The full-circuit,
  teacher-only, no-CZ, exact and matched-classical arms reproduce every persisted per-seed curve
  bit for bit. The original random stream of the *shuffled* and *random* arms could not be
  recovered, so those two arms were re-run with the repaired code
  (`outputs/mechanism_controls.json`); the first-export values are kept in
  `outputs/mechanism_controls_original_export.json` (shuffled +27.6 +/- 11.8, random +15.1 +/- 8.5
  vs. +26.7 +/- 14.4 and +14.1 +/- 3.6 now). Conclusions are unchanged.
* **QAOA per-seed files.** The first export contained per-seed part files for seeds 42-45 only;
  `outputs/qaoa_multiseed.json` (all ten seeds) is canonical. Seeds 46-51 were re-run on a
  different machine (Linux, same package versions) into `outputs/qaoa_multiseed_parts_rerun/`:
  failures found, discovery AUC and first-failure iteration are identical for all six seeds;
  the number of unique states visited differs by 1-4 on three seeds (46, 48, 50), a
  floating-point effect in COBYLA. The paper reports the canonical file.
* `code_commit` hashes inside older outputs refer to the authors' private working tree, which
  predates this public history.
* Superseded files kept for the record: `multiseed_results.json` and
  `policy_loop_comparison.json` contain with-replacement MC rows that the paper does not use
  (`mc_no_replacement.json` and `policy_loop_norepl.json` are canonical); `split_results.json`
  is the 256-shot predecessor of `split_results_1024.json`; `quantum_tree_results.json` predates
  `headline_reproducibility_seed42.json`.
* No analysis in the paper was preregistered. Table I of the paper marks which analyses were
  part of the original design, added after first results, or added in revision.

## License

MIT (see `LICENSE`).
