"""Small exploratory sweep for 8-qubit refinement variants.

This is not a final validation protocol. It is a design-space probe to identify
whether any 8D quantum refinement is worth a stricter follow-up experiment.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from phase2_qaoa.qaoa_runner import evaluate_state
from phase2_qaoa.qubo_encoder import enumerate_parameter_states
from phase3_quantum_tree.classical_layer import ClassicalTeacherModel, _normalize_state
from phase3_quantum_tree.distillation import blend
from phase3_quantum_tree.student_model import AutonomousStudentModel
from run_split_experiment import auc, cumulative_failures, state_key, stratified_split, warmstart_on_train_failures


ROOT = Path(__file__).resolve().parent
OUT_JSON = ROOT / "outputs" / "quantum_8dim_sweep.json"
SPLIT_SEEDS = list(range(42, 52))
K = 50

State = Dict[str, float]
StateKey = Tuple[float, float, float, float]


@dataclass(frozen=True)
class SweepVariant:
    name: str
    coverage: float
    angle_map: str
    entanglement: str


VARIANTS = [
    SweepVariant(f"q8_{angle}_{ent}_cov{int(cov * 100):02d}", cov, angle, ent)
    for angle in ("signed01", "absnorm", "centered")
    for ent in ("chain", "ring")
    for cov in (0.10, 0.15, 0.20, 0.25, 0.33, 0.50)
]


def labels_for(states: Sequence[State]) -> np.ndarray:
    return np.array([1.0 if evaluate_state(state)["failure"] else 0.0 for state in states], dtype=float)


def features_for(states: Sequence[State]) -> np.ndarray:
    return np.stack([_normalize_state(state) for state in states], axis=0)


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": round(float(statistics.mean(values)), 6),
        "std": round(float(statistics.pstdev(values)), 6),
    }


def theta_for(row: np.ndarray, angle_map: str) -> np.ndarray:
    row8 = row[:8]
    if angle_map == "signed01":
        return np.clip((row8 + 1.0) / 2.0, 0.0, 1.0) * np.pi
    if angle_map == "absnorm":
        scale = float(np.max(np.abs(row8))) if float(np.max(np.abs(row8))) > 0 else 1.0
        return np.clip(np.abs(row8) / scale, 0.0, 1.0) * np.pi
    if angle_map == "centered":
        return np.clip(row8, -1.0, 1.0) * (np.pi / 2.0)
    raise ValueError(f"unknown angle_map: {angle_map}")


def exact_score(qc: QuantumCircuit) -> float:
    probabilities = Statevector.from_instruction(qc).probabilities_dict()
    return float(
        np.clip(
            sum((bits.count("1") / 8.0) * float(probability) for bits, probability in probabilities.items()),
            0.0,
            1.0,
        )
    )


def q8_scores(hidden: np.ndarray, teacher_probs: np.ndarray, variant: SweepVariant) -> np.ndarray:
    n_refine = max(1, int(np.ceil(len(hidden) * variant.coverage)))
    refine_indices = set(int(index) for index in np.argsort(teacher_probs)[::-1][:n_refine].tolist())
    scores = np.zeros(len(hidden), dtype=float)
    for index, (row, teacher_prob) in enumerate(zip(hidden, teacher_probs)):
        if index not in refine_indices:
            continue
        qc = QuantumCircuit(8)
        for qubit, angle in enumerate(theta_for(row, variant.angle_map)):
            qc.ry(float(angle), qubit)
        for qubit in range(7):
            qc.cz(qubit, qubit + 1)
        if variant.entanglement == "ring":
            qc.cz(7, 0)
        qc.ry(float(np.clip(teacher_prob, 0.0, 1.0) * np.pi), 0)
        scores[index] = exact_score(qc)
    return scores


def train_student_metric(
    train_states: Sequence[State],
    train_labels: np.ndarray,
    test_states: Sequence[State],
    test_labels: np.ndarray,
    seed: int,
    train_targets: np.ndarray,
    test_targets: np.ndarray,
    test_quantum: np.ndarray,
    test_teacher: np.ndarray,
) -> Dict[str, Any]:
    x_train = features_for(train_states)
    x_test = features_for(test_states)
    student = AutonomousStudentModel(seed=seed)
    warmstart_on_train_failures(student, train_states, train_labels, {state_key(state) for state in test_states})
    student.fit(x_train, train_targets)
    scores = student.predict(x_test).reshape(-1)
    rows = []
    for index, state in enumerate(test_states):
        rows.append(
            {
                "key": state_key(state),
                "student_score": float(scores[index]),
                "target": float(test_targets[index]),
                "quantum_score": float(test_quantum[index]),
                "teacher_prob": float(test_teacher[index]),
            }
        )
    rows.sort(key=lambda row: (row["student_score"], row["target"], row["quantum_score"], row["teacher_prob"]), reverse=True)
    selected = [row["key"] for row in rows[: min(K, len(rows))]]
    failures = {state_key(state) for state, label in zip(test_states, test_labels) if int(label) == 1}
    cumulative = cumulative_failures(selected, failures)
    found = int(cumulative[-1]) if cumulative else 0
    return {
        "auc": auc(cumulative),
        "recall": round(found / len(failures), 6),
        "failures_found": found,
    }


def main() -> None:
    if OUT_JSON.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT_JSON}")
    states = enumerate_parameter_states()
    labels = labels_for(states)
    print("8D QUANTUM SWEEP")
    print(f"states={len(states)} failures={int(np.sum(labels))} variants={len(VARIANTS)} seeds={SPLIT_SEEDS}")
    per_split: List[Dict[str, Any]] = []
    for seed in SPLIT_SEEDS:
        train_indices, test_indices = stratified_split(states, labels, seed)
        train_states = [states[index] for index in train_indices]
        test_states = [states[index] for index in test_indices]
        train_labels = labels[train_indices]
        test_labels = labels[test_indices]
        x_train = features_for(train_states)
        x_test = features_for(test_states)
        teacher = ClassicalTeacherModel(seed=seed, hidden_dim=8)
        teacher.fit(x_train, train_labels.reshape(-1, 1))
        train_teacher = teacher.predict_proba(x_train).reshape(-1)
        test_teacher = teacher.predict_proba(x_test).reshape(-1)
        train_hidden = teacher.hidden_representation(x_train)
        test_hidden = teacher.hidden_representation(x_test)
        teacher_result = train_student_metric(
            train_states,
            train_labels,
            test_states,
            test_labels,
            seed,
            train_teacher,
            test_teacher,
            np.zeros(len(test_states), dtype=float),
            test_teacher,
        )
        split_record: Dict[str, Any] = {"split_seed": seed, "teacher_only_8d": teacher_result, "variants": {}}
        print(f"split_seed={seed} teacher_auc={teacher_result['auc']}")
        for variant in VARIANTS:
            train_q = q8_scores(train_hidden, train_teacher, variant)
            test_q = q8_scores(test_hidden, test_teacher, variant)
            train_targets = np.array([blend(float(p), float(q)) for p, q in zip(train_teacher, train_q)], dtype=float)
            test_targets = np.array([blend(float(p), float(q)) for p, q in zip(test_teacher, test_q)], dtype=float)
            result = train_student_metric(
                train_states,
                train_labels,
                test_states,
                test_labels,
                seed,
                train_targets,
                test_targets,
                test_q,
                test_teacher,
            )
            split_record["variants"][variant.name] = result
        per_split.append(split_record)

    aggregate: Dict[str, Any] = {}
    for variant in VARIANTS:
        lifts = [
            float(split["variants"][variant.name]["auc"]) - float(split["teacher_only_8d"]["auc"])
            for split in per_split
        ]
        aucs = [float(split["variants"][variant.name]["auc"]) for split in per_split]
        recalls = [float(split["variants"][variant.name]["recall"]) for split in per_split]
        aggregate[variant.name] = {
            "auc": mean_std(aucs),
            "recall": mean_std(recalls),
            "auc_lift_vs_teacher": mean_std(lifts),
            "positive_lift_splits": int(sum(1 for value in lifts if value > 0)),
        }
    ranked = sorted(
        aggregate.items(),
        key=lambda item: (item[1]["auc_lift_vs_teacher"]["mean"], item[1]["positive_lift_splits"]),
        reverse=True,
    )
    print("")
    print("TOP 10 BY MEAN AUC LIFT")
    for name, stats in ranked[:10]:
        print(
            f"{name}: lift_mean={stats['auc_lift_vs_teacher']['mean']} "
            f"lift_std={stats['auc_lift_vs_teacher']['std']} "
            f"auc_mean={stats['auc']['mean']} positive_splits={stats['positive_lift_splits']}"
        )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split_seeds": SPLIT_SEEDS,
        "k": K,
        "variants": [variant.__dict__ for variant in VARIANTS],
        "per_split": per_split,
        "aggregate": aggregate,
        "top_by_mean_auc_lift": [name for name, _ in ranked[:10]],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote={OUT_JSON}")


if __name__ == "__main__":
    main()
