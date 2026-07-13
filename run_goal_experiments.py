"""
Run the QTD paper-audit experiments without overwriting existing result JSON.
"""

from __future__ import annotations

import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

from phase2_qaoa.monte_carlo import random_state
from phase3_quantum_tree.pipeline import (
    evaluate_geometric_state,
    evaluate_policy_state,
    run_quantum_tree_pipeline,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
SEEDS = list(range(42, 52))
K = 50
SHOTS = 1024


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_new_json(name: str, payload: Dict[str, Any]) -> Path:
    path = OUT / name
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def auc(values: Iterable[int]) -> float:
    total = 0.0
    previous = 0.0
    for current in values:
        total += (previous + float(current)) / 2.0
        previous = float(current)
    return round(total, 3)


def time_to_first_failure(cumulative: Iterable[int]) -> Optional[int]:
    for index, value in enumerate(cumulative, start=1):
        if int(value) > 0:
            return index
    return None


def mean_std(values: List[float]) -> Dict[str, float]:
    return {
        "mean": round(float(statistics.mean(values)), 6),
        "std": round(float(statistics.pstdev(values)), 6),
    }


def metric_record(result: Dict[str, Any], elapsed: float, method: str, seed: int) -> Dict[str, Any]:
    cumulative = [int(value) for value in result["cumulative_failures"]]
    return {
        "seed": seed,
        "method": method,
        "unique_failures": int(result["total_unique_failures"]),
        "auc": auc(cumulative),
        "time_to_first_failure": time_to_first_failure(cumulative),
        "cumulative_failures": cumulative,
        "student_mse": result.get("summary", {}).get("student_mse"),
        "teacher_accuracy": result.get("summary", {}).get("teacher_accuracy"),
        "quantum_backend": result.get("summary", {}).get("quantum_backend"),
        "elapsed_seconds": round(elapsed, 3),
        "config": {
            "seed": seed,
            "k": result.get("k_iterations"),
            "shots": result.get("n_shots"),
            "candidate_pool_size": result.get("candidate_pool_size"),
            "quantum_enabled": result.get("quantum_enabled"),
            "ranking_mode": result.get("ranking_mode"),
            "evaluator": result.get("evaluator", "geometric"),
        },
    }


def run_qtd(seed: int, quantum_enabled: bool, evaluator: str) -> tuple[Dict[str, Any], float]:
    started = time.time()
    result = run_quantum_tree_pipeline(
        seed=seed,
        shots=SHOTS,
        k_iterations=K,
        quantum_enabled=quantum_enabled,
        ranking_mode="qtd" if quantum_enabled else "teacher_only",
        evaluator=evaluator,
        output_path=None,
    )
    return result, time.time() - started


def run_mc(
    seed: int,
    evaluator: Callable[[Dict[str, float]], Dict[str, Any]],
    evaluator_name: str,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    unique_failures: List[Dict[str, Any]] = []
    seen: set[str] = set()
    cumulative: List[int] = []
    started = time.time()
    for iteration in range(K):
        evaluation = evaluator(random_state(rng))
        evaluation["iteration"] = iteration
        if evaluation["failure"]:
            key = json.dumps(evaluation["params"], sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique_failures.append(evaluation)
        cumulative.append(len(unique_failures))
    return {
        "method": "monte_carlo",
        "seed": seed,
        "k_iterations": K,
        "evaluator": evaluator_name,
        "elapsed_seconds": round(time.time() - started, 3),
        "total_unique_failures": len(unique_failures),
        "cumulative_failures": cumulative,
        "auc": auc(cumulative),
        "time_to_first_failure": time_to_first_failure(cumulative),
        "failures": unique_failures,
    }


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_method: Dict[str, Dict[str, Any]] = {}
    for method in sorted({row["method"] for row in records}):
        rows = [row for row in records if row["method"] == method]
        by_method[method] = {
            "unique_failures": mean_std([float(row["unique_failures"]) for row in rows]),
            "auc": mean_std([float(row["auc"]) for row in rows]),
            "time_to_first_failure": mean_std(
                [
                    float(row["time_to_first_failure"])
                    for row in rows
                    if row["time_to_first_failure"] is not None
                ]
            ),
            "seeds_completed": [int(row["seed"]) for row in rows],
        }
    return by_method


def run_quantum_ablation_multiseed() -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    per_seed: List[Dict[str, Any]] = []
    for seed in SEEDS:
        qtd, qtd_elapsed = run_qtd(seed=seed, quantum_enabled=True, evaluator="geometric")
        teacher, teacher_elapsed = run_qtd(seed=seed, quantum_enabled=False, evaluator="geometric")
        qtd_record = metric_record(qtd, qtd_elapsed, "qtd_quantum_on", seed)
        teacher_record = metric_record(teacher, teacher_elapsed, "teacher_only_no_quantum", seed)
        records.extend([qtd_record, teacher_record])
        per_seed.append(
            {
                "seed": seed,
                "qtd_quantum_on": qtd_record,
                "teacher_only_no_quantum": teacher_record,
                "delta": {
                    "unique_failures": qtd_record["unique_failures"] - teacher_record["unique_failures"],
                    "auc": round(qtd_record["auc"] - teacher_record["auc"], 3),
                },
            }
        )
        print(
            "ablation seed "
            f"{seed}: QTD {qtd_record['unique_failures']} failures/AUC {qtd_record['auc']} | "
            f"teacher-only {teacher_record['unique_failures']} failures/AUC {teacher_record['auc']}"
        )

    deltas_failures = [float(row["delta"]["unique_failures"]) for row in per_seed]
    deltas_auc = [float(row["delta"]["auc"]) for row in per_seed]
    consistently_positive = all(value > 0 for value in deltas_failures)
    interpretation = (
        "positive_all_seeds"
        if consistently_positive
        else "within_noise_or_seed_dependent"
    )
    payload = {
        "generated_at": now_iso(),
        "question": "Does the QTD quantum layer provide a consistent lift over teacher-only ranking?",
        "config": {"seeds": SEEDS, "k": K, "shots": SHOTS, "evaluator": "geometric"},
        "per_seed": per_seed,
        "aggregate": aggregate(records),
        "delta_aggregate": {
            "unique_failures": mean_std(deltas_failures),
            "auc": mean_std(deltas_auc),
            "positive_failure_delta_seed_count": sum(1 for value in deltas_failures if value > 0),
            "positive_auc_delta_seed_count": sum(1 for value in deltas_auc if value > 0),
            "seeds_completed": len(SEEDS),
        },
        "interpretation": interpretation,
        "notes": (
            "Teacher-only disables quantum scoring and sets distilled_target=teacher_prob. "
            "AUC is trapezoidal cumulative-failure area starting from previous=0."
        ),
    }
    write_new_json("quantum_ablation_multiseed.json", payload)
    return payload


def run_reproducibility_and_fig_data() -> tuple[Dict[str, Any], Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    raw_qtd_results: List[Dict[str, Any]] = []
    for repeat in (1, 2):
        result, elapsed = run_qtd(seed=42, quantum_enabled=True, evaluator="geometric")
        raw_qtd_results.append(result)
        record = metric_record(result, elapsed, f"qtd_repeat_{repeat}", 42)
        runs.append(record)
        print(
            f"repro repeat {repeat}: failures {record['unique_failures']}, "
            f"AUC {record['auc']}, MSE {record['student_mse']}"
        )

    comparable = {
        key: [run[key] for run in runs]
        for key in ("unique_failures", "auc", "student_mse", "teacher_accuracy", "quantum_backend")
    }
    deterministic = all(len(set(map(str, values))) == 1 for values in comparable.values())
    canonical = runs[-1]
    repro_payload = {
        "generated_at": now_iso(),
        "config": {"seed": 42, "k": K, "shots": SHOTS, "evaluator": "geometric"},
        "runs": runs,
        "deterministic_for_checked_metrics": deterministic,
        "canonical_current_code": canonical,
        "paper_saved_reference": {
            "source": "outputs/quantum_tree_results.json",
            "unique_failures": 18,
            "auc": 336.0,
            "student_mse": 0.008346,
        },
        "current_code_note": (
            "Current code canon is the repeatable seed-42 rerun. The saved paper run "
            "has the same final unique failures but older cumulative ordering/AUC and MSE."
        ),
    }
    write_new_json("headline_reproducibility_seed42.json", repro_payload)

    mc = run_mc(42, evaluate_geometric_state, "geometric")
    fig_payload = {
        "generated_at": now_iso(),
        "config": {"seed": 42, "k": K, "shots": SHOTS, "evaluator": "geometric"},
        "qtd": {
            "source": "outputs/headline_reproducibility_seed42.json#/canonical_current_code",
            "cumulative_failures": raw_qtd_results[-1]["cumulative_failures"],
            "total_unique_failures": raw_qtd_results[-1]["total_unique_failures"],
            "auc": auc(raw_qtd_results[-1]["cumulative_failures"]),
            "student_mse": raw_qtd_results[-1]["summary"]["student_mse"],
        },
        "monte_carlo": {
            "source": "fresh run in run_goal_experiments.py",
            "cumulative_failures": mc["cumulative_failures"],
            "total_unique_failures": mc["total_unique_failures"],
            "auc": mc["auc"],
        },
    }
    write_new_json("fig_cumulative_data.json", fig_payload)
    return repro_payload, fig_payload


def run_policy_loop_comparison() -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    per_seed: List[Dict[str, Any]] = []
    for seed in SEEDS:
        qtd, qtd_elapsed = run_qtd(seed=seed, quantum_enabled=True, evaluator="policy")
        teacher, teacher_elapsed = run_qtd(seed=seed, quantum_enabled=False, evaluator="policy")
        mc = run_mc(seed, evaluate_policy_state, "policy")
        qtd_record = metric_record(qtd, qtd_elapsed, "qtd_quantum_on", seed)
        teacher_record = metric_record(teacher, teacher_elapsed, "teacher_only_no_quantum", seed)
        mc_record = {
            "seed": seed,
            "method": "monte_carlo",
            "unique_failures": int(mc["total_unique_failures"]),
            "auc": mc["auc"],
            "time_to_first_failure": mc["time_to_first_failure"],
            "cumulative_failures": mc["cumulative_failures"],
            "elapsed_seconds": mc["elapsed_seconds"],
            "config": {"seed": seed, "k": K, "evaluator": "policy"},
        }
        records.extend([qtd_record, teacher_record, mc_record])
        per_seed.append(
            {
                "seed": seed,
                "qtd_quantum_on": qtd_record,
                "teacher_only_no_quantum": teacher_record,
                "monte_carlo": mc_record,
            }
        )
        print(
            "policy seed "
            f"{seed}: QTD {qtd_record['unique_failures']} | "
            f"teacher-only {teacher_record['unique_failures']} | MC {mc_record['unique_failures']}"
        )

    payload = {
        "generated_at": now_iso(),
        "question": "Closed-loop discovery with Phase 1 LinearPolicy evaluator.",
        "config": {
            "seeds": SEEDS,
            "k": K,
            "shots": SHOTS,
            "evaluator": "policy",
            "policy_artifact": str(ROOT / "outputs" / "phase1" / "policy_weights.npz"),
        },
        "per_seed": per_seed,
        "aggregate": aggregate(records),
        "notes": (
            "The policy evaluator uses the saved Phase 1 LinearPolicy with greedy per-step action "
            "selection over the static four-feature encounter observation."
        ),
    }
    write_new_json("policy_loop_comparison.json", payload)
    return payload


def main() -> None:
    OUT.mkdir(exist_ok=True)
    ablation = run_quantum_ablation_multiseed()
    repro, fig_data = run_reproducibility_and_fig_data()
    policy = run_policy_loop_comparison()
    print("wrote outputs/quantum_ablation_multiseed.json")
    print("wrote outputs/headline_reproducibility_seed42.json")
    print("wrote outputs/fig_cumulative_data.json")
    print("wrote outputs/policy_loop_comparison.json")
    print(
        "quantum ablation interpretation: "
        f"{ablation['interpretation']} "
        f"(mean delta {ablation['delta_aggregate']['unique_failures']['mean']} failures, "
        f"{ablation['delta_aggregate']['auc']['mean']} AUC)"
    )
    print(
        "canonical current seed-42 QTD: "
        f"{repro['canonical_current_code']['unique_failures']} failures, "
        f"AUC {repro['canonical_current_code']['auc']}, "
        f"MSE {repro['canonical_current_code']['student_mse']}"
    )
    print(
        "figure seed-42 arrays: "
        f"QTD {fig_data['qtd']['total_unique_failures']} failures/AUC {fig_data['qtd']['auc']}; "
        f"MC {fig_data['monte_carlo']['total_unique_failures']} failures/AUC {fig_data['monte_carlo']['auc']}"
    )
    print(
        "policy-loop aggregate QTD failures mean: "
        f"{policy['aggregate']['qtd_quantum_on']['unique_failures']['mean']}"
    )


if __name__ == "__main__":
    main()
