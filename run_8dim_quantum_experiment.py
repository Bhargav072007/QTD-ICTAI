"""Exploratory 8-hidden-dimension quantum refinement ablation.

This runner keeps the leakage-free split protocol from run_split_experiment.py,
but compares an 8-dimensional classical teacher-only target against quantum
refinement variants that consume all 8 teacher hidden activations.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from phase2_qaoa.qaoa_runner import evaluate_state
from phase2_qaoa.qubo_encoder import enumerate_parameter_states
from phase3_quantum_tree.classical_layer import ClassicalTeacherModel, _normalize_state
from phase3_quantum_tree.distillation import blend
from phase3_quantum_tree.quantum_layer import QuantumDistillationLayer
from phase3_quantum_tree.student_model import AutonomousStudentModel
from run_split_experiment import auc, cumulative_failures, state_key, stratified_split, warmstart_on_train_failures


ROOT = Path(__file__).resolve().parent
OUT_JSON = ROOT / "outputs" / "quantum_8dim_experiment.json"
SPLIT_SEEDS = list(range(42, 52))
K = 50

State = Dict[str, float]
StateKey = Tuple[float, float, float, float]


def labels_for(states: Sequence[State]) -> np.ndarray:
    return np.array([1.0 if evaluate_state(state)["failure"] else 0.0 for state in states], dtype=float)


def features_for(states: Sequence[State]) -> np.ndarray:
    return np.stack([_normalize_state(state) for state in states], axis=0)


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": round(float(statistics.mean(values)), 6),
        "std": round(float(statistics.pstdev(values)), 6),
    }


@dataclass(frozen=True)
class Variant:
    name: str
    quantum_kind: str
    coverage: float = 1.0
    entanglement: str = "chain"
    teacher_injection: str = "ry_q0"
    target_mode: str = "blend"


VARIANTS = [
    Variant(name="teacher_only_8d", quantum_kind="none", target_mode="teacher"),
    Variant(name="current_3q_top25", quantum_kind="current_3q", coverage=0.25),
    Variant(name="q8_chain_top25", quantum_kind="q8_exact", coverage=0.25, entanglement="chain"),
    Variant(name="q8_chain_all", quantum_kind="q8_exact", coverage=1.0, entanglement="chain"),
    Variant(name="q8_ring_all", quantum_kind="q8_exact", coverage=1.0, entanglement="ring"),
]


def exact_hamming_score(circuit: QuantumCircuit, n_qubits: int) -> float:
    probabilities = Statevector.from_instruction(circuit).probabilities_dict()
    weighted = 0.0
    for bitstring, probability in probabilities.items():
        weighted += (bitstring.count("1") / n_qubits) * float(probability)
    return float(np.clip(weighted, 0.0, 1.0))


def q8_scores(
    hidden: np.ndarray,
    teacher_probs: np.ndarray,
    *,
    coverage: float,
    entanglement: str,
) -> np.ndarray:
    n_qubits = 8
    n_refine = max(1, int(np.ceil(len(hidden) * coverage)))
    refine_indices = set(int(index) for index in np.argsort(teacher_probs)[::-1][:n_refine].tolist())
    scores = np.zeros(len(hidden), dtype=float)
    for index, (row, teacher_prob) in enumerate(zip(hidden, teacher_probs)):
        if index not in refine_indices:
            continue
        qc = QuantumCircuit(n_qubits)
        theta = np.clip((row[:n_qubits] + 1.0) / 2.0, 0.0, 1.0) * np.pi
        for qubit, angle in enumerate(theta):
            qc.ry(float(angle), qubit)
        for qubit in range(n_qubits - 1):
            qc.cz(qubit, qubit + 1)
        if entanglement == "ring":
            qc.cz(n_qubits - 1, 0)
        qc.ry(float(np.clip(teacher_prob, 0.0, 1.0) * np.pi), 0)
        scores[index] = exact_hamming_score(qc, n_qubits)
    return scores


def quantum_scores_for(variant: Variant, hidden: np.ndarray, teacher_probs: np.ndarray, seed: int) -> np.ndarray:
    if variant.quantum_kind == "none":
        return np.zeros(len(hidden), dtype=float)
    if variant.quantum_kind == "current_3q":
        return QuantumDistillationLayer(shots=256, seed=seed).refine(hidden, teacher_probs)["quantum_scores"]
    if variant.quantum_kind == "q8_exact":
        return q8_scores(hidden, teacher_probs, coverage=variant.coverage, entanglement=variant.entanglement)
    raise ValueError(f"Unknown quantum_kind: {variant.quantum_kind}")


def run_variant(
    variant: Variant,
    split_seed: int,
    train_states: Sequence[State],
    train_labels: np.ndarray,
    test_states: Sequence[State],
    test_labels: np.ndarray,
) -> Dict[str, Any]:
    x_train = features_for(train_states)
    y_train = train_labels.reshape(-1, 1)
    x_test = features_for(test_states)

    teacher = ClassicalTeacherModel(seed=split_seed, hidden_dim=8)
    teacher.fit(x_train, y_train)
    train_probs = teacher.predict_proba(x_train).reshape(-1)
    test_probs = teacher.predict_proba(x_test).reshape(-1)
    train_hidden = teacher.hidden_representation(x_train)
    test_hidden = teacher.hidden_representation(x_test)

    train_q = quantum_scores_for(variant, train_hidden, train_probs, split_seed)
    test_q = quantum_scores_for(variant, test_hidden, test_probs, split_seed)

    if variant.target_mode == "teacher":
        train_targets = train_probs.astype(float)
        test_targets = test_probs.astype(float)
    else:
        train_targets = np.array([blend(float(p), float(q)) for p, q in zip(train_probs, train_q)], dtype=float)
        test_targets = np.array([blend(float(p), float(q)) for p, q in zip(test_probs, test_q)], dtype=float)

    student = AutonomousStudentModel(seed=split_seed)
    test_keys = {state_key(state) for state in test_states}
    warmstart_count, no_warmstart_test_overlap, warmstart_subset_train_failures = warmstart_on_train_failures(
        student,
        train_states,
        train_labels,
        test_keys,
    )
    student.fit(x_train, train_targets)
    student_scores = student.predict(x_test).reshape(-1)

    rows: List[Dict[str, Any]] = []
    for index, state in enumerate(test_states):
        rows.append(
            {
                "key": state_key(state),
                "student_score": float(student_scores[index]),
                "target": float(test_targets[index]),
                "quantum_score": float(test_q[index]),
                "teacher_prob": float(test_probs[index]),
            }
        )
    rows.sort(
        key=lambda row: (row["student_score"], row["target"], row["quantum_score"], row["teacher_prob"]),
        reverse=True,
    )

    selected_keys = [row["key"] for row in rows[: min(K, len(rows))]]
    failure_keys = {state_key(state) for state, label in zip(test_states, test_labels) if int(label) == 1}
    cumulative = cumulative_failures(selected_keys, failure_keys)
    found = int(cumulative[-1]) if cumulative else 0
    total = len(failure_keys)
    return {
        "recall": round(found / total, 6) if total else 0.0,
        "auc": auc(cumulative),
        "failures_found": found,
        "test_failures_total": total,
        "warmstart_count": warmstart_count,
        "checks": {
            "no_warmstart_test_overlap": bool(no_warmstart_test_overlap),
            "warmstart_subset_train_failures": bool(warmstart_subset_train_failures),
        },
    }


def aggregate(per_split: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_variant: Dict[str, Dict[str, Any]] = {}
    for variant in VARIANTS:
        rows = [row["variants"][variant.name] for row in per_split]
        by_variant[variant.name] = {
            "recall": mean_std([float(row["recall"]) for row in rows]),
            "auc": mean_std([float(row["auc"]) for row in rows]),
            "auc_lift_vs_teacher": mean_std(
                [
                    float(row["auc"]) - float(split["variants"]["teacher_only_8d"]["auc"])
                    for row, split in zip(rows, per_split)
                ]
            ),
            "positive_auc_lift_splits": int(
                sum(
                    1
                    for row, split in zip(rows, per_split)
                    if float(row["auc"]) > float(split["variants"]["teacher_only_8d"]["auc"])
                )
            ),
        }
    return by_variant


def main() -> None:
    if OUT_JSON.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT_JSON}")
    states = enumerate_parameter_states()
    labels = labels_for(states)
    print("8D QUANTUM VS 8D CLASSICAL EXPERIMENT")
    print(f"states={len(states)} failures={int(np.sum(labels))} k={K} split_seeds={SPLIT_SEEDS}")
    per_split: List[Dict[str, Any]] = []
    for split_seed in SPLIT_SEEDS:
        train_indices, test_indices = stratified_split(states, labels, split_seed)
        train_states = [states[index] for index in train_indices]
        test_states = [states[index] for index in test_indices]
        train_labels = labels[train_indices]
        test_labels = labels[test_indices]
        assert {state_key(state) for state in train_states}.isdisjoint({state_key(state) for state in test_states})

        split_record: Dict[str, Any] = {
            "split_seed": split_seed,
            "train_failures": int(np.sum(train_labels)),
            "test_failures": int(np.sum(test_labels)),
            "variants": {},
        }
        for variant in VARIANTS:
            split_record["variants"][variant.name] = run_variant(
                variant,
                split_seed,
                train_states,
                train_labels,
                test_states,
                test_labels,
            )
        per_split.append(split_record)
        teacher_auc = split_record["variants"]["teacher_only_8d"]["auc"]
        print(f"split_seed={split_seed} train_failures={split_record['train_failures']} test_failures={split_record['test_failures']} teacher_auc={teacher_auc}")
        for variant in VARIANTS:
            result = split_record["variants"][variant.name]
            lift = round(float(result["auc"]) - float(teacher_auc), 6)
            print(
                f"  {variant.name}: recall={result['recall']} auc={result['auc']} "
                f"lift_vs_teacher={lift} failures_found={result['failures_found']}"
            )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split_seeds": SPLIT_SEEDS,
        "k": K,
        "variants": [variant.__dict__ for variant in VARIANTS],
        "per_split": per_split,
        "aggregate": aggregate(per_split),
    }
    print("")
    print("AGGREGATE")
    for name, stats in payload["aggregate"].items():
        print(
            f"{name}: recall_mean={stats['recall']['mean']} recall_std={stats['recall']['std']} "
            f"auc_mean={stats['auc']['mean']} auc_std={stats['auc']['std']} "
            f"auc_lift_mean={stats['auc_lift_vs_teacher']['mean']} "
            f"auc_lift_std={stats['auc_lift_vs_teacher']['std']} "
            f"positive_lift_splits={stats['positive_auc_lift_splits']}"
        )
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote={OUT_JSON}")


if __name__ == "__main__":
    main()
