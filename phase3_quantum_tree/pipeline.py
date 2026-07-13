"""
Quantum Tree distillation pipeline runner.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase2_qaoa.qaoa_runner import OUT
from phase2_qaoa.qaoa_runner import evaluate_state as evaluate_geometric_state
from phase1.aviation_env_3d import ACTIONS, ROLLOUT_HORIZON, normalize_state
from phase1.train import LinearPolicy
from phase3_quantum_tree.classical_layer import run_classical_layer
from phase3_quantum_tree.distillation import build_distilled_dataset
from phase3_quantum_tree.quantum_layer import QuantumDistillationLayer
from phase3_quantum_tree.student_model import AutonomousStudentModel


Evaluator = Callable[[Dict[str, float]], Dict[str, Any]]
POLICY_ARTIFACT = ROOT / "outputs" / "phase1" / "policy_weights.npz"
_POLICY_CACHE: LinearPolicy | None = None


def _load_trained_policy() -> LinearPolicy:
    global _POLICY_CACHE
    if _POLICY_CACHE is not None:
        return _POLICY_CACHE
    if not POLICY_ARTIFACT.exists():
        raise FileNotFoundError(
            f"Missing trained policy artifact at {POLICY_ARTIFACT}. "
            "Run phase1/train.py first or use --evaluator geometric."
        )
    data = np.load(POLICY_ARTIFACT, allow_pickle=True)
    policy = LinearPolicy(seed=42)
    policy.weights = np.array(data["weights"], dtype=float)
    policy.bias = np.array(data["bias"], dtype=float)
    _POLICY_CACHE = policy
    return policy


def evaluate_policy_state(params: Dict[str, float]) -> Dict[str, Any]:
    """
    Evaluate an encounter using the saved Phase 1 LinearPolicy.

    The Phase 1 policy observes the static four-parameter encounter vector, so
    greedy per-step selection is deterministic for a fixed candidate state.
    """
    policy = _load_trained_policy()
    ego_x, ego_y, ego_z = 20.0, 100.0, 31000.0
    ego_speed = 4.5
    intruder_x = 88.0 + float(params["int_x_offset"])
    intruder_y = 10.0
    intruder_z = float(params["int_altitude"])
    intruder_heading_deg = float(params["int_heading"])
    intruder_speed = float(params["int_speed"])

    ex, ey, ez = ego_x, ego_y, ego_z
    ix, iy = intruder_x, intruder_y
    min_h_sep = float("inf")
    min_v_sep = float("inf")
    separation_loss = False
    near_miss = False
    action_trace: List[str] = []

    obs = normalize_state(params)
    for _ in range(ROLLOUT_HORIZON):
        probs = policy.action_probs(obs)
        action = ACTIONS[int(np.argmax(probs))]
        action_trace.append(action)
        if action == "climb":
            ego_heading_deg, ego_climb_rate = 0.0, 20.0
        elif action == "descend":
            ego_heading_deg, ego_climb_rate = 0.0, -20.0
        elif action == "turn_left":
            ego_heading_deg, ego_climb_rate = -18.0, 0.0
        elif action == "turn_right":
            ego_heading_deg, ego_climb_rate = 18.0, 0.0
        else:
            ego_heading_deg, ego_climb_rate = 0.0, 0.0

        ex += ego_speed * math.cos(math.radians(ego_heading_deg))
        ey += ego_speed * math.sin(math.radians(ego_heading_deg))
        ez += ego_climb_rate
        ix += intruder_speed * math.cos(math.radians(intruder_heading_deg))
        iy += intruder_speed * math.sin(math.radians(intruder_heading_deg))

        h_sep = math.hypot(ex - ix, ey - iy)
        v_sep = abs(ez - intruder_z)
        min_h_sep = min(min_h_sep, h_sep)
        min_v_sep = min(min_v_sep, v_sep)
        if h_sep < 5.0 and v_sep < 1000.0:
            separation_loss = True
        if h_sep < 7.5 and v_sep < 1500.0:
            near_miss = True

    if separation_loss:
        failure_type = "separation_loss"
    elif near_miss:
        failure_type = "near_miss"
    else:
        failure_type = "safe"

    primary_action = max(set(action_trace), key=action_trace.count)
    return {
        "params": params,
        "action": primary_action,
        "action_trace": action_trace,
        "policy_artifact": str(POLICY_ARTIFACT),
        "min_h_sep_nm": round(min_h_sep, 3),
        "min_v_sep_ft": round(min_v_sep, 1),
        "separation_loss": separation_loss,
        "near_miss": near_miss,
        "failure": failure_type in {"separation_loss", "near_miss"},
        "failure_type": failure_type,
    }


def _select_evaluator(name: str) -> Evaluator:
    if name == "geometric":
        return evaluate_geometric_state
    if name == "policy":
        return evaluate_policy_state
    raise ValueError("evaluator must be 'geometric' or 'policy'")


def _severity_from_eval(evaluation: Dict[str, Any]) -> str:
    h_sep = float(evaluation["min_h_sep_nm"])
    v_sep = float(evaluation["min_v_sep_ft"])
    if h_sep < 2.0 and v_sep < 500.0:
        return "critical"
    if h_sep < 3.0 and v_sep < 750.0:
        return "high"
    if h_sep < 5.0 and v_sep < 1000.0:
        return "medium"
    return "low"


def _scenario_alignment(state: Dict[str, float], scenario_params: Optional[Dict[str, Any]]) -> float:
    if not scenario_params:
        return 0.0
    score = 0.0
    heading = scenario_params.get("intruder_heading")
    if heading is not None:
        score -= abs(float(state["int_heading"]) - float(heading)) / 180.0
    altitude_band = scenario_params.get("altitude_band")
    if altitude_band:
        try:
            altitude_target = int(str(altitude_band).replace("FL", "")) * 100.0
            score -= abs(float(state["int_altitude"]) - altitude_target) / 10000.0
        except ValueError:
            pass
    return score


def _state_key(state: Dict[str, float]) -> Tuple[float, float, float, float]:
    return (
        float(state["int_heading"]),
        float(state["int_altitude"]),
        float(state["int_speed"]),
        float(state["int_x_offset"]),
    )


def _state_from_row(row: Dict[str, float]) -> Dict[str, float]:
    return {
        "int_heading": float(row["int_heading"]),
        "int_altitude": float(row["int_altitude"]),
        "int_speed": float(row["int_speed"]),
        "int_x_offset": float(row["int_x_offset"]),
    }


def _rank_student_candidates(
    distilled_rows: Sequence[Dict[str, float]],
    student_scores: np.ndarray,
    scenario_params: Optional[Dict[str, Any]],
) -> List[Dict[str, float]]:
    ranked_rows: List[Dict[str, float]] = []
    for index, row in enumerate(distilled_rows):
        enriched = dict(row)
        enriched["student_score"] = float(student_scores[index])
        enriched["search_score"] = float(enriched["student_score"])
        enriched["alignment_score"] = float(_scenario_alignment(enriched, scenario_params))
        ranked_rows.append(enriched)
    ranked_rows.sort(
        key=lambda row: (row["student_score"], row["distilled_target"], row["alignment_score"]),
        reverse=True,
    )
    return ranked_rows


def _primary_quantum_backend(backends: Sequence[Any]) -> str:
    for backend in backends:
        backend_name = str(backend)
        if backend_name != "skipped-low-teacher":
            return backend_name
    return str(backends[0]) if backends else "unknown"


def _select_candidate_pool(
    ranked_rows: Sequence[Dict[str, float]],
    n_generate: int,
) -> List[Dict[str, float]]:
    teacher_sorted = sorted(
        ranked_rows,
        key=lambda row: (row["teacher_prob"], row["distilled_target"], row["quantum_score"]),
        reverse=True,
    )
    return teacher_sorted[: min(len(teacher_sorted), n_generate)]


def _evaluate_student_search(
    ranked_rows: Sequence[Dict[str, float]],
    k_iterations: int,
    evaluator: Evaluator,
) -> Dict[str, Any]:
    cumulative_failures: List[int] = []
    unique_failure_keys: set[Tuple[float, float, float, float]] = set()
    top_failures: List[Dict[str, Any]] = []

    for iteration, row in enumerate(ranked_rows[:k_iterations]):
        state = _state_from_row(row)
        evaluation = evaluator(state)
        scenario_type = evaluation["failure_type"]

        if scenario_type in {"separation_loss", "near_miss"}:
            unique_failure_keys.add(_state_key(state))

        cumulative_failures.append(len(unique_failure_keys))

    inspection_budget = min(len(ranked_rows), max(k_iterations, 25))
    for iteration, row in enumerate(ranked_rows[:inspection_budget]):
        state = _state_from_row(row)
        evaluation = evaluator(state)
        scenario_type = evaluation["failure_type"]
        if scenario_type not in {"separation_loss", "near_miss"}:
            continue
        top_failures.append(
            {
                "type": scenario_type,
                "severity": _severity_from_eval(evaluation),
                "h_sep_nm": evaluation["min_h_sep_nm"],
                "v_sep_ft": evaluation["min_v_sep_ft"],
                "description": (
                    f"Intruder at {int(state['int_altitude']):.0f} ft, "
                    f"crossing heading {int(state['int_heading']):.0f} deg"
                ),
                "distilled_target": round(float(row["distilled_target"]), 6),
                "teacher_prob": round(float(row["teacher_prob"]), 6),
                "quantum_score": round(float(row["quantum_score"]), 6),
                "student_score": round(float(row["student_score"]), 6),
                "scenario_type": scenario_type,
                "iteration": iteration,
                "action": evaluation.get("action"),
            }
        )

    top_failures.sort(
        key=lambda item: (
            {"critical": 3, "high": 2, "medium": 1, "low": 0}.get(item["severity"], 0),
            item["student_score"],
            item["distilled_target"],
        ),
        reverse=True,
    )
    total_unique_failures = len(unique_failure_keys)
    failure_rate = round(total_unique_failures / max(k_iterations, 1), 6)
    return {
        "cumulative_failures": cumulative_failures,
        "total_unique_failures": total_unique_failures,
        "failures_found": total_unique_failures,
        "failure_rate": failure_rate,
        "top_failures": top_failures[:10],
    }


def warmstart_student(
    student: AutonomousStudentModel,
    failure_states_path: Path = Path("outputs/failure_states.json"),
    seed: int = 42,
) -> int:
    """
    Pre-trains the student on known failure states from Phase 1.
    Returns the number of failure states used for warmstart.

    This reduces cold-start lag by giving the student prior knowledge of
    failure-inducing parameter regions before distillation training.
    Uses failure_states.json produced by phase1/evaluate.py.
    """
    if not failure_states_path.exists():
        return 0

    import json
    payload = json.loads(failure_states_path.read_text(encoding="utf-8"))
    rows = payload.get("failure_states", []) if isinstance(payload, dict) else []
    if not rows:
        return 0

    from phase3_quantum_tree.classical_layer import _normalize_state
    warmstart_features = []
    warmstart_targets = []
    for row in rows:
        params = row.get("params")
        if not isinstance(params, dict):
            continue
        feat = _normalize_state(params)
        warmstart_features.append(feat)
        warmstart_targets.append(1.0)  # all are known failures

    if not warmstart_features:
        return 0

    import numpy as np
    x_ws = np.stack(warmstart_features, axis=0)
    y_ws = np.array(warmstart_targets, dtype=float)
    # Brief warmstart pass (50 epochs) before full distillation training
    orig_epochs = student.epochs
    student.epochs = 50
    student.fit(x_ws, y_ws)
    student.epochs = orig_epochs
    return len(warmstart_features)


def run_quantum_tree_pipeline(
    seed: int = 42,
    shots: int = 1024,
    scenario_params: Optional[Dict[str, Any]] = None,
    fast: bool = False,
    k_iterations: Optional[int] = None,
    quantum_enabled: bool = True,
    quantum_variant: str = "full",
    ranking_mode: str = "qtd",
    evaluator: str = "geometric",
    output_path: Optional[str | Path] = "outputs/",
) -> Dict[str, Any]:
    if ranking_mode not in {"qtd", "teacher_only"}:
        raise ValueError("ranking_mode must be 'qtd' or 'teacher_only'")
    if not quantum_enabled and ranking_mode == "qtd":
        ranking_mode = "teacher_only"

    k_iterations = int(k_iterations) if k_iterations is not None else (5 if fast else 50)
    effective_shots = 256 if fast else shots
    evaluator_fn = _select_evaluator(evaluator)
    classical = run_classical_layer(seed=seed, evaluator=evaluator_fn)
    if quantum_enabled:
        quantum_layer = QuantumDistillationLayer(shots=effective_shots, seed=seed, variant=quantum_variant)
        quantum = quantum_layer.refine(classical["hidden"], classical["teacher_probs"])
        no_quantum_mode = None
    else:
        quantum = {
            "backend": np.array(["quantum-disabled"] * len(classical["teacher_probs"]), dtype=object),
            "quantum_scores": np.zeros(len(classical["teacher_probs"]), dtype=float),
        }
        no_quantum_mode = "teacher_probability_target_quantum_score_forced_zero"
    distilled_rows = build_distilled_dataset(
        features=classical["features"],
        states=classical["states"],
        labels=classical["labels"],
        teacher_probs=classical["teacher_probs"],
        quantum_scores=quantum["quantum_scores"],
    )
    if ranking_mode == "teacher_only":
        for row in distilled_rows:
            row["distilled_target"] = float(row["teacher_prob"])

    student = AutonomousStudentModel(seed=seed)
    failure_path = ROOT / "outputs" / "failure_states.json"
    n_warmstart = warmstart_student(student, failure_states_path=failure_path, seed=seed)
    # n_warmstart > 0 means student was pre-trained on known failure states
    targets = np.array([row["distilled_target"] for row in distilled_rows], dtype=float)
    student_metrics = student.fit(classical["features"], targets)
    student_preds = student.predict(classical["features"]).reshape(-1)
    ranked_rows = _rank_student_candidates(distilled_rows, student_preds, scenario_params)
    n_evaluate = min(k_iterations, len(ranked_rows))
    n_generate = min(len(ranked_rows), max(n_evaluate, n_evaluate * 3))
    candidate_pool = _select_candidate_pool(ranked_rows, n_generate=n_generate)
    candidate_pool.sort(
        key=lambda row: (row["student_score"], row["distilled_target"], row["alignment_score"]),
        reverse=True,
    )
    evaluation_rows = candidate_pool[:n_evaluate]
    search_results = _evaluate_student_search(evaluation_rows, len(evaluation_rows), evaluator=evaluator_fn)
    ranked = evaluation_rows[:10]
    summary = {
        "teacher_accuracy": classical["teacher_metrics"]["accuracy"],
        "teacher_loss": classical["teacher_metrics"]["loss"],
        "quantum_backend": _primary_quantum_backend(quantum["backend"]),
        "mean_quantum_score": round(float(np.mean(quantum["quantum_scores"])), 6),
        "mean_teacher_prob": round(float(np.mean(classical["teacher_probs"])), 6),
        "mean_student_score": round(float(np.mean(student_preds)), 6),
        "student_mse": student_metrics["mse"],
    }
    results = {
        "run_id": f"qt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "architecture": [
            "Layer 1: classical teacher simulation with backpropagation",
            "Layer 2: quantum refinement layer using IBM Qiskit-compatible sampling",
            "Layer 3: distilled dataset used to train an autonomous student model",
        ],
        "scenario_params": scenario_params or {},
        "evaluator": evaluator,
        "fast_mode": fast,
        "quantum_enabled": bool(quantum_enabled),
        "quantum_variant": quantum_variant if quantum_enabled else None,
        "ranking_mode": ranking_mode,
        "no_quantum_mode": no_quantum_mode,
        "teacher_accuracy": float(summary["teacher_accuracy"]),
        "quantum_backend": str(summary["quantum_backend"]),
        "student_mse": float(summary["student_mse"]),
        "failures_found": int(search_results["failures_found"]),
        "failure_rate": float(search_results["failure_rate"]),
        "k_iterations": n_evaluate,
        "n_shots": effective_shots,
        "n_scenarios": n_evaluate,
        "candidate_pool_size": n_generate,
        "candidate_pool_rule": "top teacher_prob rows before final student ranking",
        "cumulative_failures": search_results["cumulative_failures"],
        "total_unique_failures": int(search_results["total_unique_failures"]),
        "summary": summary,
        "top_failures": search_results["top_failures"],
        "top_distilled_states": ranked,
        "n_warmstart_states": n_warmstart,
    }

    effective_output_path = None if fast else output_path
    if effective_output_path is not None:
        output_root = Path(effective_output_path)
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "quantum_tree_results.json").write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the QAA Quantum Tree distillation pipeline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--fast", action="store_true")
    quantum_group = parser.add_mutually_exclusive_group()
    quantum_group.add_argument("--quantum-enabled", dest="quantum_enabled", action="store_true", default=True)
    quantum_group.add_argument("--no-quantum", dest="quantum_enabled", action="store_false")
    parser.add_argument("--ranking-mode", choices=["qtd", "teacher_only"], default="qtd")
    parser.add_argument("--evaluator", choices=["geometric", "policy"], default="geometric")
    args = parser.parse_args()
    results = run_quantum_tree_pipeline(
        seed=args.seed,
        shots=args.shots,
        fast=args.fast,
        k_iterations=args.k,
        quantum_enabled=args.quantum_enabled,
        ranking_mode=args.ranking_mode,
        evaluator=args.evaluator,
    )
    print("Quantum Tree distillation complete")
    print(f"Teacher accuracy : {results['summary']['teacher_accuracy']}")
    print(f"Quantum backend  : {results['summary']['quantum_backend']}")
    print(f"Student MSE      : {results['summary']['student_mse']}")
  