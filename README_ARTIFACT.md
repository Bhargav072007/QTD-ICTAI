# Anonymous reproduction artifact

Offline source and persisted JSON evidence.

## Setup

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.lock
```

## Reproduce

```bash
python run_mc_no_replacement_analysis.py
python run_split_experiment.py --shots 1024 --output-name split_results_1024.json
python run_mechanism_controls.py
python run_qaoa_multiseed.py --seed 42
python run_qaoa_multiseed.py --assemble
python run_traceability_checks.py
```

Repeat QAOA per seed before assembly.
