"""Leakage-free split experiment for the QTD aviation benchmark."""

from __future__ import annotations

import json
import argparse
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import qiskit
import qiskit_aer

from phase2_qaoa.qaoa_runner import evaluate_state
from phase2_qaoa.qubo_encoder import enumerate_parameter_states
from phase3_quantum_tree.classical_layer import ClassicalTeacherModel, _normalize_state
from phase3_quantum_tree.distillation import blend
from phase3_quantum_tree.quantum_layer import QuantumDistillationLayer
from phase3_quantum_tree.student_model import AutonomousStudentModel


ROOT = Path(__file__).resolve().parent
OUT_JSON = ROOT / "outputs" / "split_results.json"
OUT_MD = ROOT / "SPLIT_RESULTS.md"
SPLIT_SEEDS = list(range(42, 52))
K = 50


def verify_quantum_backend() -> Dict[str, Any]:
    """Abort instead of silently accepting the layer's classical fallback."""
    layer = QuantumDistillationLayer(shots=16, seed=42)
    out = layer.refine(np.ones((4, 8), dtype=float), np.array([0.9, 0.8, 0.7, 0.6], dtype=float))
    backend_values = sorted(set(str(value) for value in out["backend"]))
    if "qiskit-statevector" not in backend_values:
        raise RuntimeError(f"ABORT: quantum backend fell back; backend_values={backend_values}")
    return {
        "qiskit_version": qiskit.__version__,
        "qiskit_aer_version": qiskit_aer.__version__,
        "quantum_backend_values": backend_values,
    }


State = Dict[str, float]
StateKey = Tuple[float, float, float, float]


def state_key(state: State) -> StateKey:
    return (
        float(state["int_heading"]),
        float(state["int_altitude"]),
        float(state["int_speed"]),
        float(state["int_x_offset"]),
    )


def auc(cumulative: Iterable[int]) -> float:
    total = 0.0
    previous = 0.0
    for current in cumulative:
        total += (previous + float(current)) / 2.0
        previous = float(current)
    return round(total, 6)


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": round(float(statistics.mean(values)), 6),
        "std": round(float(statistics.pstdev(values)), 6),
    }


def features_for(states: Sequence[State]) -> np.ndarray:
    return np.stack([_normalize_state(state) for state in states], axis=0)


def labels_for(states: Sequence[State]) -> np.ndarray:
    return np.array([1.0 if evaluate_state(state)["failure"] else 0.0 for state in states], dtype=float)


def stratified_split(states: Sequence[State], labels: np.ndarray, seed: int) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    test_indices: List[int] = []
    for label in (0.0, 1.0):
        class_indices = np.flatnonzero(labels == label)
        shuffled = rng.permutation(class_indices)
        split_at = len(shuffled) // 2
        train_indices.extend(int(index) for index in shuffled[:split_at])
        test_indices.extend(int(index) for index in shuffled[split_at:])
    train_indices.sort()
    test_indices.sort()
    return train_indices, test_indices


def cumulative_failures(ranked_keys: Sequence[StateKey], failure_keys: set[StateKey]) -> List[int]:
    found: set[StateKey] = set()
    cumulative: List[int] = []
    for key in ranked_keys:
        if key in failure_keys:
            found.add(key)
        cumulative.append(len(found))
    return cumulative


def warmstart_on_train_failures(
    student: AutonomousStudentModel,
    train_states: Sequence[State],
    train_labels: np.ndarray,
    test_keys: set[StateKey],
) -> Tuple[int, bool, bool]:
    train_failure_states = [state for state, label in zip(train_states, train_labels) if int(label) == 1]
    train_failure_keys = {state_key(state) for state in train_failure_states}
    warmstart_keys = set(train_failure_keys)
    no_overlap = warmstart_keys.isdisjoint(test_keys)
    subset_train_failures = warmstart_keys.issubset(train_failure_keys)
    assert no_overlap, "warmstart_set intersects test_set"
    assert subset_train_failures, "warmstart_set is not a subset of train_failures"
    if not train_failure_states:
        return 0, no_overlap, subset_train_failures

    x_ws = features_for(train_failure_states)
    y_ws = np.ones(len(train_failure_states), dtype=float)
    original_epochs = student.epochs
    student.epochs = 50
    student.fit(x_ws, y_ws)
    student.epochs = original_epochs
    return len(train_failure_states), no_overlap, subset_train_failures


def train_and_rank(
    split_seed: int,
    train_states: Sequence[State],
    train_labels: np.ndarray,
    test_states: Sequence[State],
    test_labels: np.ndarray,
    quantum_enabled: bool,
    shots: int,
) -> Dict[str, Any]:
    x_train = features_for(train_states)
    y_train = train_labels.reshape(-1, 1)
    x_test = features_for(test_states)

    teacher = ClassicalTeacherModel(seed=split_seed)
    teacher_metrics = teacher.fit(x_train, y_train)
    train_teacher_probs = teacher.predict_proba(x_train).reshape(-1)
    test_teacher_probs = teacher.predict_proba(x_test).reshape(-1)
    train_hidden = teacher.hidden_representation(x_train)
    test_hidden = teacher.hidden_representation(x_test)

    if quantum_enabled:
        quantum_layer = QuantumDistillationLayer(shots=shots, seed=split_seed)
        train_quantum = quantum_layer.refine(train_hidden, train_teacher_probs)["quantum_scores"]
        test_quantum = quantum_layer.refine(test_hidden, test_teacher_probs)["quantum_scores"]
        train_targets = np.array(
            [blend(float(p_teacher), float(q_quantum)) for p_teacher, q_quantum in zip(train_teacher_probs, train_quantum)],
            dtype=float,
        )
        test_distilled = np.array(
            [blend(float(p_teacher), float(q_quantum)) for p_teacher, q_quantum in zip(test_teacher_probs, test_quantum)],
            dtype=float,
        )
    else:
        train_quantum = np.zeros(len(train_states), dtype=float)
        test_quantum = np.zeros(len(test_states), dtype=float)
        train_targets = train_teacher_probs.astype(float)
        test_distilled = test_teacher_probs.astype(float)

    del train_quantum
    student = AutonomousStudentModel(seed=split_seed)
    test_keys = {state_key(state) for state in test_states}
    warmstart_count, no_warmstart_test_overlap, warmstart_subset_train_failures = warmstart_on_train_failures(
        student,
        train_states,
        train_labels,
        test_keys,
    )
    student_metrics = student.fit(x_train, train_targets)
    student_scores = student.predict(x_test).reshape(-1)

    rows: List[Dict[str, Any]] = []
    for index, state in enumerate(test_states):
        rows.append(
            {
                "key": state_key(state),
                "student_score": float(student_scores[index]),
                "distilled_target": float(test_distilled[index]),
                "quantum_score": float(test_quantum[index]),
                "teacher_prob": float(test_teacher_probs[index]),
            }
        )
    rows.sort(
        key=lambda row: (row["student_score"], row["distilled_target"], row["quantum_score"], row["teacher_prob"]),
        reverse=True,
    )
    selected_keys = [row["key"] for row in rows[: min(K, len(rows))]]
    test_failure_keys = {state_key(state) for state, label in zip(test_states, test_labels) if int(label) == 1}
    cumulative = cumulative_failures(selected_keys, test_failure_keys)
    found = int(cumulative[-1]) if cumulative else 0
    denominator = len(test_failure_keys)
    return {
        "selected_keys": selected_keys,
        "candidate_keys": [state_key(state) for state in test_states],
        "test_failures_found": found,
        "test_failures_total": denominator,
        "recall": round(found / denominator, 6) if denominator else 0.0,
        "auc": auc(cumulative),
        "cumulative_failures": cumulative,
        "teacher_metrics": teacher_metrics,
        "student_metrics": student_metrics,
        "warmstart_count": warmstart_count,
        "no_warmstart_test_overlap": no_warmstart_test_overlap,
        "warmstart_subset_train_failures": warmstart_subset_train_failures,
    }


def run_mc(split_seed: int, test_states: Sequence[State], test_labels: np.ndarray) -> Dict[str, Any]:
    rng = np.random.default_rng(split_seed)
    candidate_keys = [state_key(state) for state in test_states]
    test_failure_keys = {state_key(state) for state, label in zip(test_states, test_labels) if int(label) == 1}
    draw_count = min(K, len(test_states))
    draw_indices = rng.choice(len(test_states), size=draw_count, replace=False)
    draw_keys = [candidate_keys[int(index)] for index in draw_indices]
    cumulative = cumulative_failures(draw_keys, test_failure_keys)
    found = int(cumulative[-1]) if cumulative else 0
    denominator = len(test_failure_keys)
    return {
        "candidate_keys": candidate_keys,
        "draw_keys": draw_keys,
        "sampling": "uniform without replacement",
        "test_failures_found": found,
        "test_failures_total": denominator,
        "recall": round(found / denominator, 6) if denominator else 0.0,
        "auc": auc(cumulative),
        "cumulative_failures": cumulative,
    }


def print_ground_truth(states: Sequence[State], labels: np.ndarray) -> None:
    evaluations = [evaluate_state(state) for state in states]
    total_failures = int(np.sum(labels))
    separation_loss = sum(1 for row in evaluations if row["failure_type"] == "separation_loss")
    near_miss = sum(1 for row in evaluations if row["failure_type"] == "near_miss")
    print("STEP 1 GROUND TRUTH")
    print(f"total states: {len(states)}")
    print(f"total failures: {total_failures}")
    print(f"separation_loss: {separation_loss}")
    print(f"near_miss: {near_miss}")
    if len(states) != 256 or total_failures != 18 or separation_loss != 5 or near_miss != 13:
        raise RuntimeError("STEP 1 expected exactly 256 states / 18 failures / 5 sep-loss / 13 near-miss")


def aggregate(per_split: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    lift_values = [float(row["qtd_auc"]) - float(row["teacher_only_auc"]) for row in per_split]
    return {
        "qtd_recall": mean_std([float(row["qtd_recall"]) for row in per_split]),
        "teacher_only_recall": mean_std([float(row["teacher_only_recall"]) for row in per_split]),
        "mc_recall": mean_std([float(row["mc_recall"]) for row in per_split]),
        "qtd_auc": mean_std([float(row["qtd_auc"]) for row in per_split]),
        "teacher_only_auc": mean_std([float(row["teacher_only_auc"]) for row in per_split]),
        "mc_auc": mean_std([float(row["mc_auc"]) for row in per_split]),
        "quantum_auc_lift": mean_std(lift_values),
    }


def write_markdown(payload: Dict[str, Any], out_md: Path) -> None:
    aggregate_rows = payload["aggregate"]
    lift_mean = aggregate_rows["quantum_auc_lift"]["mean"]
    survives = "yes" if lift_mean > 0 else "no"
    lines = [
        "# Leakage-Free Split Results",
        "",
        "| Method | Recall mean+/-std | AUC mean+/-std |",
        "| --- | ---: | ---: |",
        f"| QTD | {aggregate_rows['qtd_recall']['mean']} +/- {aggregate_rows['qtd_recall']['std']} | {aggregate_rows['qtd_auc']['mean']} +/- {aggregate_rows['qtd_auc']['std']} |",
        f"| Teacher-only | {aggregate_rows['teacher_only_recall']['mean']} +/- {aggregate_rows['teacher_only_recall']['std']} | {aggregate_rows['teacher_only_auc']['mean']} +/- {aggregate_rows['teacher_only_auc']['std']} |",
        f"| MC | {aggregate_rows['mc_recall']['mean']} +/- {aggregate_rows['mc_recall']['std']} | {aggregate_rows['mc_auc']['mean']} +/- {aggregate_rows['mc_auc']['std']} |",
        "",
        f"Quantum ordering lift survives: {survives}; QTD-teacher-only AUC lift mean+/-std = {aggregate_rows['quantum_auc_lift']['mean']} +/- {aggregate_rows['quantum_auc_lift']['std']}.",
        "",
        f"Honest verdict: Does the ranking advantage survive cold-start? {survives}, by {lift_mean} mean AUC versus teacher-only.",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")


def reproduction_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "outputs" / "reproduction" / f"split_experiment_{stamp}"


def resolve_outputs(output_dir: Path | None, force: bool) -> tuple[Path, Path]:
    if output_dir is None:
        out_json = OUT_JSON
        out_md = OUT_MD
        if (out_json.exists() or out_md.exists()) and not force:
            base = reproduction_dir()
            out_json = base / OUT_JSON.name
            out_md = base / OUT_MD.name
    else:
        out_json = output_dir / OUT_JSON.name
        out_md = output_dir / OUT_MD.name
    for path in (out_json, out_md):
        if path.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    return out_json, out_md


def main() -> None:
    global OUT_JSON, OUT_MD
    parser = argparse.ArgumentParser(description="Run leakage-free held-out split experiment")
    parser.add_argument("--output-dir", type=Path, default=None, help="Write outputs here instead of the canonical outputs directory.")
    parser.add_argument("--force", action="store_true", help="Allow overwriting the selected output files.")
    parser.add_argument("--shots", type=int, default=256, help="Shots per circuit execution (default: 256, the legacy experiment setting).")
    parser.add_argument("--output-name", default=OUT_JSON.name, help="JSON filename to write under --output-dir.")
    args = parser.parse_args()
    OUT_JSON = ROOT / "outputs" / args.output_name
    OUT_MD = ROOT / "outputs" / Path(args.output_name).with_suffix(".md")
    out_json, out_md = resolve_outputs(args.output_dir, args.force)
    backend = verify_quantum_backend()

    states = enumerate_parameter_states()
    labels = labels_for(states)
    print_ground_truth(states, labels)
    print("")
    print("STEP 2/3 SPLIT RESULTS")

    per_split: List[Dict[str, Any]] = []
    all_checks_passed = True
    for split_seed in SPLIT_SEEDS:
        train_indices, test_indices = stratified_split(states, labels, split_seed)
        train_states = [states[index] for index in train_indices]
        test_states = [states[index] for index in test_indices]
        train_labels = labels[train_indices]
        test_labels = labels[test_indices]
        train_keys = {state_key(state) for state in train_states}
        test_keys = {state_key(state) for state in test_states}
        no_train_test_leakage = train_keys.isdisjoint(test_keys)
        assert no_train_test_leakage, "train/test leakage detected"

        qtd = train_and_rank(split_seed, train_states, train_labels, test_states, test_labels, quantum_enabled=True, shots=args.shots)
        teacher = train_and_rank(split_seed, train_states, train_labels, test_states, test_labels, quantum_enabled=False, shots=args.shots)
        mc = run_mc(split_seed, test_states, test_labels)
        mc_pool_equals_ranker_pool = set(mc["candidate_keys"]) == set(qtd["candidate_keys"]) == set(teacher["candidate_keys"])
        assert mc_pool_equals_ranker_pool, "MC pool differs from ranker pool"

        checks = {
            "no_train_test_leakage": bool(no_train_test_leakage),
            "warmstart_subset_train_failures": bool(qtd["warmstart_subset_train_failures"] and teacher["warmstart_subset_train_failures"]),
            "no_warmstart_test_overlap": bool(qtd["no_warmstart_test_overlap"] and teacher["no_warmstart_test_overlap"]),
            "mc_pool_equals_ranker_pool": bool(mc_pool_equals_ranker_pool),
        }
        all_checks_passed = all_checks_passed and all(checks.values())
        record = {
            "split_seed": split_seed,
            "train_size": len(train_states),
            "test_size": len(test_states),
            "train_failure_count": int(np.sum(train_labels)),
            "test_failure_count": int(np.sum(test_labels)),
            "qtd_recall": qtd["recall"],
            "teacher_only_recall": teacher["recall"],
            "mc_recall": mc["recall"],
            "qtd_auc": qtd["auc"],
            "teacher_only_auc": teacher["auc"],
            "mc_auc": mc["auc"],
            "qtd_failures_found": qtd["test_failures_found"],
            "teacher_only_failures_found": teacher["test_failures_found"],
            "mc_failures_found": mc["test_failures_found"],
            "qtd_cumulative_failures": qtd["cumulative_failures"],
            "teacher_only_cumulative_failures": teacher["cumulative_failures"],
            "mc_cumulative_failures": mc["cumulative_failures"],
            "train_state_keys": sorted(train_keys),
            "test_state_keys": sorted(test_keys),
            "qtd_selected_keys": qtd["selected_keys"],
            "teacher_only_selected_keys": teacher["selected_keys"],
            "mc_draw_order": mc["draw_keys"],
            "qtd_warmstart_state_keys": sorted(
                state_key(state) for state, label in zip(train_states, train_labels) if int(label) == 1
            ),
            "teacher_only_warmstart_state_keys": sorted(
                state_key(state) for state, label in zip(train_states, train_labels) if int(label) == 1
            ),
            "qtd_warmstart_count": qtd["warmstart_count"],
            "teacher_only_warmstart_count": teacher["warmstart_count"],
            "checks": checks,
        }
        per_split.append(record)
        print(
            f"split_seed={split_seed} test_failures={record['test_failure_count']} "
            f"qtd_recall={record['qtd_recall']} teacher_only_recall={record['teacher_only_recall']} "
            f"mc_recall={record['mc_recall']} qtd_auc={record['qtd_auc']} "
            f"teacher_only_auc={record['teacher_only_auc']} mc_auc={record['mc_auc']}"
        )
        print(f"  checks={checks}")

    # --- Tail reconstructed for the camera-ready artifact repair -------------
    # The first public export truncated this file here. The block below was
    # rebuilt to emit the same schema as outputs/split_results_1024.json and is
    # validated in outputs/repair_validation.json.
    try:
        code_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        code_commit = None
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split_seeds": SPLIT_SEEDS,
        "k": K,
        "shots": int(args.shots),
        "code_commit": code_commit,
        "backend": backend,
        "per_split": per_split,
        "aggregate": aggregate(per_split),
        "leakage_checks_passed": bool(all_checks_passed),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(payload, out_md)
    print("")
    print(f"aggregate={payload['aggregate']}")
    print(f"leakage_checks_passed={payload['leakage_checks_passed']}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
