"""
Cold-start QTD (budget-matched), NEW EXPERIMENT.

The headline QTD pipeline fits the teacher on all 256 labelled grid states and
warm-starts the student on known failures before ranking. The paper discloses
this supervision asymmetry; this script converts that disclosure into an
experiment by running QTD with NO full-grid teacher fit and NO warm-start.

Protocol (per seed, geometric evaluator):
  1. Evaluate 5 uniform-random states (counted against the 50-eval budget).
  2. Loop until exactly 50 unique states have been evaluated:
       - Refit the teacher (4 -> 8 tanh -> 1) on ALL states evaluated so far,
         re-ranking the remaining candidates. To bound runtime the refit (and the
         quantum + student re-ranking) happens once every 5 evaluations; between
         refits the next-best remaining candidate from the cached ranking is used.
       - Layer 2: standard quantum refinement over the top-25% of the *remaining*
         candidates by current teacher probability p_T.
       - Layer 3: student trained on the distilled targets of the remaining pool.
       - Evaluate the single top-ranked not-yet-evaluated candidate and append its
         true label to the training set.

Variants: coldstart_qtd (quantum on) and coldstart_teacher_only (q_s == 0).
Seeds 42-51.

Output: outputs/coldstart_qtd.json (per-seed records + aggregate + quantum deltas).

Reproduce:
    python run_coldstart_qtd.py
"""

from __future__ import annotations

import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import qiskit
import qiskit_aer

from phase2_qaoa.qaoa_runner import evaluate_state
from phase2_qaoa.qubo_encoder import PARAM_GRID, enumerate_parameter_states
from phase3_quantum_tree.classical_layer import ClassicalTeacherModel, _normalize_state
from phase3_quantum_tree.distillation import blend
from phase3_quantum_tree.quantum_layer import QuantumDistillationLayer
from phase3_quantum_tree.student_model import AutonomousStudentModel


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
SEEDS = list(range(42, 52))
K = 50
N_INITIAL = 5
REFIT_INTERVAL = 5
SHOTS = 1024
TOTAL_FAILURES = 18


def auc(cumulative) -> float:
    total = 0.0
    previous = 0.0
    for current in cumulative:
        total += (previous + float(current)) / 2.0
        previous = float(current)
    return round(total, 6)


def first_failure(cumulative) -> Optional[int]:
    for index, value in enumerate(cumulative, start=1):
        if int(value) > 0:
            return index
    return None


def mean_std(values) -> Dict[str, Any]:
    values = list(values)
    if not values:
        return {"mean": None, "std": None}
    return {
        "mean": round(float(statistics.mean(values)), 6),
        "std": round(float(statistics.pstdev(values)), 6),
    }


def _random_unique_indices(rng: np.random.Generator, n: int, total: int) -> List[int]:
    """Pick n unique grid indices by uniform PARAM_GRID draw (matches MC style)."""
    lengths = [len(PARAM_GRID[name]) for name in PARAM_GRID]
    chosen: List[int] = []
    seen: set[int] = set()
    while len(chosen) < n:
        multi = tuple(int(rng.integers(0, length)) for length in lengths)
        # Row-major flat index into enumerate_parameter_states ordering.
        flat = ((multi[0] * lengths[1] + multi[1]) * lengths[2] + multi[2]) * lengths[3] + multi[3]
        if flat in seen:
            continue
        seen.add(flat)
        chosen.append(flat)
    return chosen


def _rank_remaining(
    features_all: np.ndarray,
    evaluated: Dict[int, float],
    remaining: List[int],
    quantum_enabled: bool,
    seed: int,
) -> Tuple[List[int], set]:
    """Refit the teacher on evaluated states and rank the remaining candidates."""
    eval_indices = sorted(evaluated.keys())
    x_eval = features_all[eval_indices]
    y_eval = np.array([evaluated[i] for i in eval_indices], dtype=float).reshape(-1, 1)

    teacher = ClassicalTeacherModel(seed=seed)
    teacher.fit(x_eval, y_eval)

    x_rem = features_all[remaining]
    teacher_probs = teacher.predict_proba(x_rem).reshape(-1)
    hidden = teacher.hidden_representation(x_rem)

    backend_values: set = set()
    if quantum_enabled:
        refined = QuantumDistillationLayer(shots=SHOTS, seed=seed).refine(hidden, teacher_probs)
        quantum_scores = refined["quantum_scores"]
        backend_values.update(str(value) for value in refined["backend"])
        targets = np.array(
            [blend(float(p), float(q)) for p, q in zip(teacher_probs, quantum_scores)],
            dtype=float,
        )
    else:
        quantum_scores = np.zeros(len(remaining), dtype=float)
        targets = teacher_probs.astype(float)

    student = AutonomousStudentModel(seed=seed)
    student.fit(x_rem, targets)
    student_scores = student.predict(x_rem).reshape(-1)

    order = sorted(
        range(len(remaining)),
        key=lambda j: (student_scores[j], targets[j], teacher_probs[j]),
        reverse=True,
    )
    ranked_indices = [remaining[j] for j in order]
    return ranked_indices, backend_values


def run_coldstart_seed(seed: int, quantum_enabled: bool) -> Dict[str, Any]:
    states = enumerate_parameter_states()
    features_all = np.stack([_normalize_state(state) for state in states], axis=0)

    rng = np.random.default_rng(seed)
    evaluated: Dict[int, float] = {}
    found = 0
    cumulative: List[int] = []
    backend_values: set = set()
    started = time.time()

    # Step 1: 5 uniform-random evaluated states (counted against budget).
    for flat in _random_unique_indices(rng, N_INITIAL, len(states)):
        label = 1.0 if evaluate_state(states[flat])["failure"] else 0.0
        evaluated[flat] = label
        if label > 0:
            found += 1
        cumulative.append(found)

    ranked: List[int] = []
    since_refit = REFIT_INTERVAL  # force a refit on the first selection
    while len(evaluated) < K:
        remaining = [i for i in range(len(states)) if i not in evaluated]
        if not remaining:
            break
        if since_refit >= REFIT_INTERVAL or not ranked:
            ranked, refit_backend = _rank_remaining(
                features_all, evaluated, remaining, quantum_enabled, seed
            )
            backend_values.update(refit_backend)
            since_refit = 0

        # Take the best not-yet-evaluated candidate from the cached ranking.
        next_index = None
        while ranked:
            candidate = ranked.pop(0)
            if candidate not in evaluated:
                next_index = candidate
                break
        if next_index is None:
            ranked, refit_backend = _rank_remaining(
                features_all, evaluated, remaining, quantum_enabled, seed
            )
            backend_values.update(refit_backend)
            since_refit = 0
            next_index = ranked.pop(0)

        label = 1.0 if evaluate_state(states[next_index])["failure"] else 0.0
        evaluated[next_index] = label
        if label > 0:
            found += 1
        cumulative.append(found)
        since_refit += 1

    quantum_backend = "classical-teacher-only"
    if quantum_enabled:
        quantum_backend = "qiskit-statevector" if "qiskit-statevector" in backend_values else "MISSING"

    return {
        "seed": seed,
        "variant": "coldstart_qtd" if quantum_enabled else "coldstart_teacher_only",
        "unique_failures": found,
        "recall": round(found / TOTAL_FAILURES, 6),
        "auc": auc(cumulative),
        "time_to_first_failure": first_failure(cumulative),
        "cumulative_failures": cumulative,
        "quantum_backend": quantum_backend,
        "quantum_backend_values": sorted(backend_values),
        "elapsed_seconds": round(time.time() - started, 3),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    per_seed: List[Dict[str, Any]] = []
    all_backend_values: set = set()

    for seed in SEEDS:
        qtd = run_coldstart_seed(seed, quantum_enabled=True)
        teacher_only = run_coldstart_seed(seed, quantum_enabled=False)
        all_backend_values.update(qtd["quantum_backend_values"])
        if qtd["quantum_backend"] != "qiskit-statevector":
            raise RuntimeError(
                f"ABORT: cold-start QTD seed {seed} did not use qiskit-statevector "
                f"(backend={qtd['quantum_backend']}, values={qtd['quantum_backend_values']})"
            )
        quantum_delta = {
            "unique_failures": qtd["unique_failures"] - teacher_only["unique_failures"],
            "auc": round(qtd["auc"] - teacher_only["auc"], 6),
            "time_to_first_failure": (
                None
                if qtd["time_to_first_failure"] is None
                or teacher_only["time_to_first_failure"] is None
                else qtd["time_to_first_failure"] - teacher_only["time_to_first_failure"]
            ),
        }
        per_seed.append(
            {
                "seed": seed,
                "coldstart_qtd": qtd,
                "coldstart_teacher_only": teacher_only,
                "quantum_delta": quantum_delta,
            }
        )
        print(
            f"seed {seed}: QTD {qtd['unique_failures']}f/AUC {qtd['auc']} | "
            f"teacher-only {teacher_only['unique_failures']}f/AUC {teacher_only['auc']} | "
            f"delta_auc {quantum_delta['auc']}"
        )

    def _agg(variant: str) -> Dict[str, Any]:
        rows = [row[variant] for row in per_seed]
        firsts = [r["time_to_first_failure"] for r in rows if r["time_to_first_failure"] is not None]
        return {
            "unique_failures": mean_std([r["unique_failures"] for r in rows]),
            "recall": mean_std([r["recall"] for r in rows]),
            "auc": mean_std([r["auc"] for r in rows]),
            "time_to_first_failure": mean_std(firsts),
            "first_failure_excluded_zero_failure_seeds": sum(
                1 for r in rows if r["time_to_first_failure"] is None
            ),
        }

    delta_failures = [row["quantum_delta"]["unique_failures"] for row in per_seed]
    delta_auc = [row["quantum_delta"]["auc"] for row in per_seed]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "Cold-start QTD (budget-matched)",
        "backend": {
            "qiskit_version": qiskit.__version__,
            "qiskit_aer_version": qiskit_aer.__version__,
            "quantum_backend_values": sorted(all_backend_values),
        },
        "config": {
            "seeds": SEEDS,
            "k": K,
            "evaluator": "geometric",
            "cold_start": True,
            "full_grid_teacher_fit": False,
            "warm_start_on_failures": False,
            "initial_random_states": N_INITIAL,
            "refit_interval": REFIT_INTERVAL,
            "shots": SHOTS,
            "total_states": 256,
            "total_failures": TOTAL_FAILURES,
            "architecture": "teacher 4->8(tanh)->1; quantum Layer-2 top-25% of remaining; student logistic on distilled targets",
        },
        "per_seed": per_seed,
        "aggregate": {
            "coldstart_qtd": _agg("coldstart_qtd"),
            "coldstart_teacher_only": _agg("coldstart_teacher_only"),
            "quantum_delta": {
                "unique_failures": mean_std(delta_failures),
                "auc": mean_std(delta_auc),
                "positive_failure_delta_seed_count": sum(1 for v in delta_failures if v > 0),
                "positive_auc_delta_seed_count": sum(1 for v in delta_auc if v > 0),
            },
        },
    }
    out_path = OUT / "coldstart_qtd.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("")
    print("AGGREGATE (mean+/-pstd)")
    print(f"coldstart_qtd unique_failures  : {payload['aggregate']['coldstart_qtd']['unique_failures']}")
    print(f"coldstart_qtd auc              : {payload['aggregate']['coldstart_qtd']['auc']}")
    print(f"coldstart_teacher_only failures: {payload['aggregate']['coldstart_teacher_only']['unique_failures']}")
    print(f"coldstart_teacher_only auc     : {payload['aggregate']['coldstart_teacher_only']['auc']}")
    print(f"quantum_delta auc              : {payload['aggregate']['quantum_delta']['auc']}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
