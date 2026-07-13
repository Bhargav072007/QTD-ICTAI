"""
Adaptive classical baseline: cross-entropy method (CEM) over the 4x4x4x4 grid.

This is a cold-start adaptive search baseline that does NOT see the full labeled
grid before ranking. It maintains an independent categorical distribution per
parameter dimension (initialised uniform), samples batches of not-yet-evaluated
states from the product distribution, evaluates them with the requested
environment evaluator, and updates each categorical toward the elite states.

It is purely classical (no quantum layer is invoked), so runs produced here do
not carry a quantum_backend provenance tag; the classical fallback discussion in
the README does not apply. The evaluation budget follows the same
"uniform-without-replacement" convention as outputs/mc_no_replacement.json:
exactly K unique environment evaluations, none repeated.
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from phase2_qaoa.qubo_encoder import PARAM_GRID

Evaluator = Callable[[Dict[str, float]], Dict[str, Any]]
StateKey = Tuple[float, float, float, float]

PARAM_ORDER: Tuple[str, ...] = tuple(PARAM_GRID.keys())

# Fixed CEM hyperparameters (not tuned per seed; recorded in the output JSON).
BATCH_SIZE = 10
ELITE_FRACTION = 0.2
ALPHA_KEEP_OLD = 0.7
NEAR_MISS_H = 7.5
NEAR_MISS_V = 1500.0


def auc(cumulative: Sequence[int]) -> float:
    """Trapezoidal cumulative-failure AUC starting from previous=0."""
    total = 0.0
    previous = 0.0
    for current in cumulative:
        total += (previous + float(current)) / 2.0
        previous = float(current)
    return round(total, 6)


def first_failure(cumulative: Sequence[int]) -> Optional[int]:
    """1-based index of the first evaluation with a discovered failure."""
    for index, value in enumerate(cumulative, start=1):
        if int(value) > 0:
            return index
    return None


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": round(float(statistics.mean(values)), 6),
        "std": round(float(statistics.pstdev(values)), 6),
    }


def _state_key(state: Dict[str, float]) -> StateKey:
    return (
        float(state["int_heading"]),
        float(state["int_altitude"]),
        float(state["int_speed"]),
        float(state["int_x_offset"]),
    )


def _severity(evaluation: Dict[str, Any]) -> float:
    """Continuous severity score; higher means a closer (more severe) encounter."""
    h_sep = float(evaluation["min_h_sep_nm"])
    v_sep = float(evaluation["min_v_sep_ft"])
    return -((h_sep / NEAR_MISS_H) + (v_sep / NEAR_MISS_V))


def _sample_state(
    probs: Dict[str, np.ndarray], rng: np.random.Generator
) -> Dict[str, float]:
    return {
        name: float(rng.choice(PARAM_GRID[name], p=probs[name]))
        for name in PARAM_ORDER
    }


def _sample_new_batch(
    probs: Dict[str, np.ndarray],
    evaluated: set[StateKey],
    all_keys: List[StateKey],
    key_to_state: Dict[StateKey, Dict[str, float]],
    batch_size: int,
    rng: np.random.Generator,
) -> List[Dict[str, float]]:
    """
    Draw up to ``batch_size`` not-yet-evaluated states from the product
    distribution. Rejection-samples from the categorical product; if the current
    distribution is too concentrated to surface a new state within the attempt
    budget, falls back to a uniform draw over the remaining states so the run
    always reaches the full evaluation budget.
    """
    batch: List[Dict[str, float]] = []
    batch_keys: set[StateKey] = set()
    max_attempts = batch_size * 200
    attempts = 0
    while len(batch) < batch_size and attempts < max_attempts:
        state = _sample_state(probs, rng)
        key = _state_key(state)
        attempts += 1
        if key in evaluated or key in batch_keys:
            continue
        batch.append(state)
        batch_keys.add(key)

    if len(batch) < batch_size:
        remaining = [
            key for key in all_keys if key not in evaluated and key not in batch_keys
        ]
        rng.shuffle(remaining)
        for key in remaining:
            if len(batch) >= batch_size:
                break
            batch.append(key_to_state[key])
            batch_keys.add(key)
    return batch


def _update_distribution(
    probs: Dict[str, np.ndarray],
    elites: Sequence[Dict[str, float]],
    alpha: float,
) -> Dict[str, np.ndarray]:
    if not elites:
        return probs
    updated: Dict[str, np.ndarray] = {}
    for name in PARAM_ORDER:
        values = PARAM_GRID[name]
        empirical = np.zeros(len(values), dtype=float)
        for state in elites:
            empirical[values.index(float(state[name]))] += 1.0
        empirical /= float(len(elites))
        new_probs = alpha * probs[name] + (1.0 - alpha) * empirical
        new_probs = new_probs / new_probs.sum()
        updated[name] = new_probs
    return updated


def run_cem_seed(
    seed: int,
    evaluator: Evaluator,
    evaluator_name: str,
    total_failures: int,
    k: int = 50,
    batch_size: int = BATCH_SIZE,
    elite_fraction: float = ELITE_FRACTION,
    alpha: float = ALPHA_KEEP_OLD,
) -> Dict[str, Any]:
    """Run a single-seed cross-entropy-method search and record its trajectory."""
    rng = np.random.default_rng(seed)

    # Enumerate the full grid once so rejection sampling can fall back cleanly.
    states: List[Dict[str, float]] = []
    for heading in PARAM_GRID["int_heading"]:
        for altitude in PARAM_GRID["int_altitude"]:
            for speed in PARAM_GRID["int_speed"]:
                for offset in PARAM_GRID["int_x_offset"]:
                    states.append(
                        {
                            "int_heading": float(heading),
                            "int_altitude": float(altitude),
                            "int_speed": float(speed),
                            "int_x_offset": float(offset),
                        }
                    )
    key_to_state = {_state_key(state): state for state in states}
    all_keys = list(key_to_state.keys())

    probs: Dict[str, np.ndarray] = {
        name: np.full(len(PARAM_GRID[name]), 1.0 / len(PARAM_GRID[name]), dtype=float)
        for name in PARAM_ORDER
    }

    evaluated: set[StateKey] = set()
    found: set[StateKey] = set()
    cumulative: List[int] = []
    evaluation_order: List[Dict[str, Any]] = []
    started = time.time()

    while len(evaluated) < k:
        batch = _sample_new_batch(
            probs, evaluated, all_keys, key_to_state, batch_size, rng
        )
        if not batch:
            break

        batch_records: List[Dict[str, Any]] = []
        for state in batch:
            if len(evaluated) >= k:
                break
            key = _state_key(state)
            evaluation = evaluator(state)
            is_failure = bool(evaluation["failure"])
            evaluated.add(key)
            if is_failure:
                found.add(key)
            cumulative.append(len(found))
            record = {
                "eval_index": len(evaluated),
                "params": {name: float(state[name]) for name in PARAM_ORDER},
                "failure": is_failure,
                "failure_type": evaluation.get("failure_type"),
                "min_h_sep_nm": evaluation.get("min_h_sep_nm"),
                "min_v_sep_ft": evaluation.get("min_v_sep_ft"),
                "severity": round(_severity(evaluation), 6),
            }
            evaluation_order.append(record)
            batch_records.append(record)

        # Elite set: failures in this batch, else the top fraction by severity.
        elite_states: List[Dict[str, float]] = [
            key_to_state[
                (
                    rec["params"]["int_heading"],
                    rec["params"]["int_altitude"],
                    rec["params"]["int_speed"],
                    rec["params"]["int_x_offset"],
                )
            ]
            for rec in batch_records
            if rec["failure"]
        ]
        if not elite_states and batch_records:
            ranked = sorted(batch_records, key=lambda r: r["severity"], reverse=True)
            elite_count = max(1, int(round(elite_fraction * len(batch_records))))
            elite_states = [
                key_to_state[
                    (
                        rec["params"]["int_heading"],
                        rec["params"]["int_altitude"],
                        rec["params"]["int_speed"],
                        rec["params"]["int_x_offset"],
                    )
                ]
                for rec in ranked[:elite_count]
            ]
        probs = _update_distribution(probs, elite_states, alpha)

    unique_failures = len(found)
    return {
        "seed": seed,
        "k": len(evaluated),
        "sampling": "CEM adaptive without replacement",
        "evaluator": evaluator_name,
        "unique_failures": unique_failures,
        "recall": round(unique_failures / total_failures, 6) if total_failures else 0.0,
        "auc": auc(cumulative),
        "time_to_first_failure": first_failure(cumulative),
        "cumulative_failures": cumulative,
        "final_distribution": {
            name: [round(float(value), 6) for value in probs[name]]
            for name in PARAM_ORDER
        },
        "elapsed_seconds": round(time.time() - started, 3),
        "evaluation_order": evaluation_order,
    }


def aggregate_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    firsts = [
        float(row["time_to_first_failure"])
        for row in records
        if row["time_to_first_failure"] is not None
    ]
    return {
        "unique_failures": mean_std([float(row["unique_failures"]) for row in records]),
        "recall": mean_std([float(row["recall"]) for row in records]),
        "auc": mean_std([float(row["auc"]) for row in records]),
        "time_to_first_failure": mean_std(firsts) if firsts else {"mean": None, "std": None},
        "first_failure_excluded_zero_failure_seeds": int(
            sum(1 for row in records if row["time_to_first_failure"] is None)
        ),
    }
