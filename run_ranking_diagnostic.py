"""
Ranking diagnostic for the iter-17 anomaly, INSTRUMENTATION.

Headline QTD (seed 42) discovers its first failure only at iteration 17 despite
full-grid supervision. This script reproduces the exact headline ranking
(seed 42, k=50, candidate pool 150, geometric evaluator) using the very same
pipeline helpers, then dumps the first 50 evaluated candidates in order with all
intermediate scores, plus where each of the 18 true failures sits in the 150-pool
ranking. It does this for both the QTD (quantum-on) and teacher-only variants.

The point is to verify the *mechanism*: the teacher assigns its highest
probabilities to a cluster of non-failing states, so the top ranks are consumed
by teacher false positives before any true failure surfaces.

Output: outputs/ranking_diagnostic_seed42.json

Reproduce:
    python run_ranking_diagnostic.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from phase3_quantum_tree.classical_layer import run_classical_layer
from phase3_quantum_tree.distillation import build_distilled_dataset
from phase3_quantum_tree.pipeline import (
    ROOT,
    _rank_student_candidates,
    _select_candidate_pool,
    _state_from_row,
    _state_key,
    evaluate_geometric_state,
    warmstart_student,
)
from phase3_quantum_tree.quantum_layer import QuantumDistillationLayer
from phase3_quantum_tree.student_model import AutonomousStudentModel


OUT = ROOT / "outputs"
SEED = 42
SHOTS = 1024
K = 50
POOL = 150


def _warmstart_keys() -> set:
    path = OUT / "failure_states.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("failure_states", []) if isinstance(payload, dict) else []
    keys = set()
    for row in rows:
        params = row.get("params")
        if isinstance(params, dict):
            keys.add(_state_key(params))
    return keys


def _auc(cumulative: List[int]) -> float:
    total = 0.0
    previous = 0.0
    for current in cumulative:
        total += (previous + float(current)) / 2.0
        previous = float(current)
    return round(total, 6)


def _first_failure(cumulative: List[int]) -> Optional[int]:
    for index, value in enumerate(cumulative, start=1):
        if int(value) > 0:
            return index
    return None


def diagnose(quantum_enabled: bool, warmstart_keys: set) -> Dict[str, Any]:
    """Replicate the headline pipeline ranking and record every candidate score."""
    classical = run_classical_layer(seed=SEED, evaluator=evaluate_geometric_state)
    if quantum_enabled:
        quantum = QuantumDistillationLayer(shots=SHOTS, seed=SEED).refine(
            classical["hidden"], classical["teacher_probs"]
        )
        quantum_scores = quantum["quantum_scores"]
        backend_values = sorted(set(str(value) for value in quantum["backend"]))
    else:
        quantum_scores = np.zeros(len(classical["teacher_probs"]), dtype=float)
        backend_values = ["quantum-disabled"]

    distilled_rows = build_distilled_dataset(
        features=classical["features"],
        states=classical["states"],
        labels=classical["labels"],
        teacher_probs=classical["teacher_probs"],
        quantum_scores=quantum_scores,
    )
    if not quantum_enabled:
        for row in distilled_rows:
            row["distilled_target"] = float(row["teacher_prob"])

    student = AutonomousStudentModel(seed=SEED)
    n_warmstart = warmstart_student(
        student, failure_states_path=OUT / "failure_states.json", seed=SEED
    )
    targets = np.array([row["distilled_target"] for row in distilled_rows], dtype=float)
    student.fit(classical["features"], targets)
    student_preds = student.predict(classical["features"]).reshape(-1)

    ranked_rows = _rank_student_candidates(distilled_rows, student_preds, None)
    candidate_pool = _select_candidate_pool(ranked_rows, n_generate=POOL)
    candidate_pool.sort(
        key=lambda row: (row["student_score"], row["distilled_target"], row["alignment_score"]),
        reverse=True,
    )
    evaluation_rows = candidate_pool[:K]

    # True labels + warm-start membership for the whole pool (150) in student order.
    def _row_record(rank: int, row: Dict[str, Any]) -> Dict[str, Any]:
        state = _state_from_row(row)
        key = _state_key(state)
        true_failure = bool(evaluate_geometric_state(state)["failure"])
        return {
            "rank": rank,
            "params": {name: float(state[name]) for name in state},
            "teacher_prob": round(float(row["teacher_prob"]), 6),
            "quantum_score": round(float(row["quantum_score"]), 6),
            "distilled_target": round(float(row["distilled_target"]), 6),
            "student_score": round(float(row["student_score"]), 6),
            "true_failure": true_failure,
            "in_warmstart_set": key in warmstart_keys,
        }

    evaluated_order = [_row_record(i + 1, row) for i, row in enumerate(evaluation_rows)]
    pool_ranking = [_row_record(i + 1, row) for i, row in enumerate(candidate_pool)]

    cumulative: List[int] = []
    found: set = set()
    for row in evaluation_rows:
        state = _state_from_row(row)
        if bool(evaluate_geometric_state(state)["failure"]):
            found.add(_state_key(state))
        cumulative.append(len(found))

    # Where the 18 true failures sit in the 150-pool student ranking.
    pool_key_to_rank = {
        _state_key(_state_from_row(row)): i + 1 for i, row in enumerate(candidate_pool)
    }
    failure_positions: List[Dict[str, Any]] = []
    for row in ranked_rows:
        state = _state_from_row(row)
        if not bool(evaluate_geometric_state(state)["failure"]):
            continue
        key = _state_key(state)
        rank_in_pool = pool_key_to_rank.get(key)
        failure_positions.append(
            {
                "params": {name: float(state[name]) for name in state},
                "rank_in_pool_150": rank_in_pool,
                "in_pool_150": rank_in_pool is not None,
                "evaluated_in_top50": rank_in_pool is not None and rank_in_pool <= K,
                "teacher_prob": round(float(row["teacher_prob"]), 6),
                "quantum_score": round(float(row["quantum_score"]), 6),
                "distilled_target": round(float(row["distilled_target"]), 6),
                "student_score": round(float(row["student_score"]), 6),
                "in_warmstart_set": key in warmstart_keys,
            }
        )
    failure_positions.sort(
        key=lambda item: (item["rank_in_pool_150"] is None, item["rank_in_pool_150"] or 10**9)
    )

    top16 = evaluated_order[:16]
    n_top16_teacher_false_positives = sum(
        1 for record in top16 if record["teacher_prob"] > 0.5 and not record["true_failure"]
    )
    best_failure = failure_positions[0] if failure_positions else None
    student_score_16th_eval = (
        round(float(evaluated_order[15]["student_score"]), 6) if len(evaluated_order) >= 16 else None
    )

    return {
        "variant": "qtd" if quantum_enabled else "teacher_only",
        "quantum_backend_values": backend_values,
        "n_warmstart_states": n_warmstart,
        "config": {
            "seed": SEED,
            "k": K,
            "candidate_pool_size": POOL,
            "shots": SHOTS,
            "evaluator": "geometric",
            "quantum_enabled": quantum_enabled,
        },
        "cumulative_failures": cumulative,
        "total_unique_failures": int(cumulative[-1]) if cumulative else 0,
        "auc": _auc(cumulative),
        "time_to_first_failure": _first_failure(cumulative),
        "summary": {
            "n_top16_teacher_false_positives": int(n_top16_teacher_false_positives),
            "n_true_failures_in_top16": int(sum(1 for r in top16 if r["true_failure"])),
            "n_top16_in_warmstart_set": int(sum(1 for r in top16 if r["in_warmstart_set"])),
            "n_true_failures_total": int(sum(1 for f in failure_positions)),
            "n_failures_in_pool_150": int(sum(1 for f in failure_positions if f["in_pool_150"])),
            "n_failures_in_top50": int(sum(1 for f in failure_positions if f["evaluated_in_top50"])),
            "best_failure_rank_in_pool_150": (
                best_failure["rank_in_pool_150"] if best_failure else None
            ),
            "best_failure_student_score": (
                best_failure["student_score"] if best_failure else None
            ),
            "best_failure_teacher_prob": (
                best_failure["teacher_prob"] if best_failure else None
            ),
            "best_failure_quantum_score": (
                best_failure["quantum_score"] if best_failure else None
            ),
            "best_failure_distilled_target": (
                best_failure["distilled_target"] if best_failure else None
            ),
            "student_score_16th_evaluated": student_score_16th_eval,
            "max_teacher_prob_among_true_failures": round(
                max((f["teacher_prob"] for f in failure_positions), default=0.0), 6
            ),
            "max_teacher_prob_among_top16": round(
                max((r["teacher_prob"] for r in top16), default=0.0), 6
            ),
            "min_teacher_prob_among_top16": round(
                min((r["teacher_prob"] for r in top16), default=0.0), 6
            ),
        },
        "evaluated_order_first_50": evaluated_order,
        "pool_ranking_150": pool_ranking,
        "failure_positions": failure_positions,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    warmstart_keys = _warmstart_keys()

    qtd = diagnose(quantum_enabled=True, warmstart_keys=warmstart_keys)
    teacher_only = diagnose(quantum_enabled=False, warmstart_keys=warmstart_keys)

    if qtd["time_to_first_failure"] != 17 or qtd["total_unique_failures"] != 18:
        raise RuntimeError(
            "ABORT: replication does not match headline "
            f"(first_failure={qtd['time_to_first_failure']}, total={qtd['total_unique_failures']}; "
            "expected 17 / 18). Not overwriting diagnostic."
        )
    if qtd["quantum_backend_values"] and "qiskit-statevector" not in qtd["quantum_backend_values"]:
        raise RuntimeError(f"ABORT: quantum fell back: {qtd['quantum_backend_values']}")

    s = qtd["summary"]
    explanation = (
        f"All 16 top-ranked candidates the pipeline evaluates first are truly safe, and the "
        f"first true failure sits at rank {s['best_failure_rank_in_pool_150']} in the student ranking, edged out because the "
        f"16th safe state scores {s['student_score_16th_evaluated']} versus the best failure's {s['best_failure_student_score']}. "
        f"The root cause is the teacher's compressed, weakly-discriminative probabilities: it assigns "
        f"no true failure a probability above {s['max_teacher_prob_among_true_failures']} while giving safe states probabilities up to "
        f"{s['max_teacher_prob_among_top16']} (only {s['n_top16_teacher_false_positives']} of the top 16 even exceeds 0.5), so the distilled targets and the "
        f"student that learns them place those safe states first. "
        f"The quantum layer lifts the first failure's quantum_score to {s['best_failure_quantum_score']}, but the "
        f"confidence-adaptive blend caps quantum influence for low-teacher-probability states, so its distilled "
        f"target only reaches {s['best_failure_distilled_target']} and its student score stays just below the safe cluster, so the "
        f"first failure is not evaluated until iteration {qtd['time_to_first_failure']}."
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "question": "Why does headline QTD (seed 42) find its first failure only at iteration 17?",
        "warmstart_set_size": len(warmstart_keys),
        "qtd": qtd,
        "teacher_only": teacher_only,
        "mechanism_explanation": explanation,
    }
    out_path = OUT / "ranking_diagnostic_seed42.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("QTD variant:")
    print(f"  first_failure_iter={qtd['time_to_first_failure']} total_failures={qtd['total_unique_failures']} auc={qtd['auc']}")
    print(f"  n_top16_teacher_false_positives={qtd['summary']['n_top16_teacher_false_positives']}")
    print(f"  best_failure_rank_in_pool_150={qtd['summary']['best_failure_rank_in_pool_150']}")
    print(f"  n_failures_in_pool_150={qtd['summary']['n_failures_in_pool_150']} n_failures_in_top50={qtd['summary']['n_failures_in_top50']}")
    print("Teacher-only variant:")
    print(f"  first_failure_iter={teacher_only['time_to_first_failure']} total_failures={teacher_only['total_unique_failures']} auc={teacher_only['auc']}")
    print(f"  n_top16_teacher_false_positives={teacher_only['summary']['n_top16_teacher_false_positives']}")
    print("")
    print("MECHANISM:")
    print(explanation)
    print("")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
