"""Controlled QTD mechanism ablation with deterministic, per-seed provenance."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import qiskit
import qiskit_aer

from phase3_quantum_tree.pipeline import run_quantum_tree_pipeline
from phase3_quantum_tree.quantum_layer import QuantumDistillationLayer
from run_mc_no_replacement_analysis import exact_tests


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
SEEDS = list(range(42, 52))
K = 50
SHOTS = 1024
ARMS = {
    "qtd_full": "full",
    "teacher_only": None,
    "no_cz": "no_cz",
    "analytic": "analytic",
    "matched_classical": "matched_classical",
    "shuffled": "shuffled",
    "random": "random",
}


def verify_quantum_backend() -> Dict[str, Any]:
    layer = QuantumDistillationLayer(shots=16, seed=42, variant="full")
    result = layer.refine(np.ones((4, 8), dtype=float), np.array([0.9, 0.8, 0.7, 0.6]))
    backends = sorted(set(map(str, result["backend"])))
    if "qiskit-statevector" not in backends:
        raise RuntimeError(f"ABORT: quantum backend fell back; backend_values={backends}")
    return {
        "qiskit_version": qiskit.__version__,
        "qiskit_aer_version": qiskit_aer.__version__,
        "quantum_backend_values": backends,
    }


def auc(cumulative: Iterable[int]) -> float:
    previous = 0.0
    total = 0.0
    for current in cumulative:
        total += (previous + float(current)) / 2.0
        previous = float(current)
    return round(total, 6)


def first_failure(cumulative: Sequence[int]) -> int | None:
    return next((index for index, value in enumerate(cumulative, start=1) if int(value) > 0), None)


def mean_pstd(values: Sequence[float]) -> Dict[str, float]:
    return {"mean": round(float(statistics.mean(values)), 6), "std": round(float(statistics.pstdev(values)), 6)}


def checked_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    return {key: row[key] for key in ("unique_failures", "auc", "first_failure", "cumulative_failures")}


def run_arm(seed: int, arm: str) -> Dict[str, Any]:
    variant = ARMS[arm]
    if variant is None:
        result = run_quantum_tree_pipeline(
            seed=seed, shots=SHOTS, k_iterations=K, quantum_enabled=False,
            ranking_mode="teacher_only", evaluator="geometric", output_path=None,
        )
    else:
        result = run_quantum_tree_pipeline(
            seed=seed, shots=SHOTS, k_iterations=K, quantum_enabled=True,
            quantum_variant=variant, ranking_mode="qtd", evaluator="geometric", output_path=None,
        )
    cumulative = [int(value) for value in result["cumulative_failures"]]
    return {
        "seed": seed,
        "unique_failures": int(result["total_unique_failures"]),
        "auc": auc(cumulative),
        "first_failure": first_failure(cumulative),
        "cumulative_failures": cumulative,
        "quantum_backend": result["summary"]["quantum_backend"],
        "quantum_variant": result.get("quantum_variant"),
    }


def bootstrap_mean_ci(values: Sequence[float]) -> Dict[str, float]:
    values_np = np.asarray(values, dtype=float)
    rng = np.random.default_rng(0)
    samples = rng.choice(values_np, size=(10_000, len(values_np)), replace=True).mean(axis=1)
    return {
        "mean": round(float(values_np.mean()), 6),
        "ci_95_low": round(float(np.quantile(samples, 0.025)), 6),
        "ci_95_high": round(float(np.quantile(samples, 0.975)), 6),
        "resamples": 10_000,
        "seed": 0,
    }


def safe_exact_tests(values: Sequence[float]) -> Dict[str, Any]:
    """Report the exact-test convention explicitly when all paired deltas are zero."""
    if not any(float(value) != 0.0 for value in values):
        return {
            "wilcoxon_p": None,
            "sign_test_p": None,
            "positive_count": 0,
            "negative_count": 0,
            "zero_count": len(values),
        }
    return exact_tests(values)


def paired_summary(arm: str, reference: str, rows: Dict[str, list[Dict[str, Any]]]) -> Dict[str, Any]:
    auc_deltas = [round(float(a["auc"]) - float(b["auc"]), 6) for a, b in zip(rows[arm], rows[reference])]
    failure_deltas = [float(a["unique_failures"]) - float(b["unique_failures"]) for a, b in zip(rows[arm], rows[reference])]
    return {
        "reference": reference,
        "auc_delta": {
            "per_seed": auc_deltas,
            "aggregate": mean_pstd(auc_deltas),
            "exact_tests": safe_exact_tests(auc_deltas),
            "bootstrap_mean_95_ci": bootstrap_mean_ci(auc_deltas),
        },
        "unique_failure_delta": {
            "per_seed": failure_deltas,
            "aggregate": mean_pstd(failure_deltas),
            "exact_tests": safe_exact_tests(failure_deltas),
            "bootstrap_mean_95_ci": bootstrap_mean_ci(failure_deltas),
        },
    }


def load_existing_full() -> Dict[int, Dict[str, Any]]:
    path = OUT / "quantum_ablation_multiseed.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["seed"]): row["qtd_quantum_on"] for row in data["per_seed"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QTD mechanism controls")
    parser.add_argument("--force", action="store_true", help="Allow replacement of mechanism_controls.json.")
    args = parser.parse_args()
    output = OUT / "mechanism_controls.json"
    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing canonical output: {output}")

    backend = verify_quantum_backend()
    rows: Dict[str, list[Dict[str, Any]]] = {arm: [] for arm in ARMS}
    deterministic: Dict[str, Dict[str, bool]] = {arm: {} for arm in ARMS}
    for seed in SEEDS:
        for arm in ARMS:
            first = run_arm(seed, arm)
            second = run_arm(seed, arm)
            same = checked_metrics(first) == checked_metrics(second)
            deterministic[arm][str(seed)] = same
            if not same:
                raise RuntimeError(f"ABORT: non-deterministic checked metrics for {arm}, seed {seed}")
            rows[arm].append(first)
            print(f"seed={seed} arm={arm} failures={first['unique_failures']} auc={first['auc']} deterministic={same}")

    previous = load_existing_full()
    sanity = {
        str(seed): checked_metrics(rows["qtd_full"][index]) == {
            "unique_failures": int(previous[seed]["unique_failures"]),
            "auc": float(previous[seed]["auc"]),
            "first_failure": previous[seed]["time_to_first_failure"],
            "cumulative_failures": previous[seed]["cumulative_failures"],
        }
        for index, seed in enumerate(SEEDS)
    }
    if not all(sanity.values()):
        raise RuntimeError(f"ABORT: qtd_full does not reproduce quantum_ablation_multiseed.json: {sanity}")

    arm_payload: Dict[str, Any] = {}
    for arm, arm_rows in rows.items():
        arm_payload[arm] = {
            "per_seed": arm_rows,
            "aggregate": {
                "unique_failures": mean_pstd([float(row["unique_failures"]) for row in arm_rows]),
                "auc": mean_pstd([float(row["auc"]) for row in arm_rows]),
                "first_failure": mean_pstd([float(row["first_failure"]) for row in arm_rows if row["first_failure"] is not None]),
            },
        }
    comparisons = {
        arm: {
            "vs_teacher_only": paired_summary(arm, "teacher_only", rows),
            "vs_qtd_full": paired_summary(arm, "qtd_full", rows),
        }
        for arm in ARMS
    }
    interpretation = (
        "The file contains controlled replacements for CZ gates, shot sampling, and the circuit score. "
        "Interpret only the paired AUC deltas and their bootstrap intervals; a control reproduces the full-arm lift "
        "only when its delta versus teacher-only overlaps the full-arm delta and its delta versus qtd_full is near zero."
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "backend": backend,
        "config": {"seeds": SEEDS, "k": K, "pool": 150, "shots": SHOTS, "evaluator": "geometric", "top_gate_fraction": 0.25},
        "determinism_checked": deterministic,
        "qtd_full_sanity_matches_quantum_ablation_multiseed": sanity,
        "arms": arm_payload,
        "paired_comparisons": comparisons,
        "interpretation": interpretation,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
