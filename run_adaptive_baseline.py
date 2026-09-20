"""
Run the adaptive cross-entropy-method (CEM) classical baseline.

Motivation: the paper's most likely reviewer objection is "you only beat uniform
Monte Carlo". This script adds a *cold-start adaptive* classical baseline that
learns where failures are as it goes, using the same 50-unique-evaluation budget
and the same shared metric functions as the Monte-Carlo comparator, so the QTD
advantage is measured against a competitive search rather than a naive one.

Outputs (schema mirrors outputs/mc_no_replacement.json):
    outputs/adaptive_baseline.json         (geometric evaluator, denominator 18)
    outputs/adaptive_baseline_policy.json   (policy evaluator, denominator 12)

Reproduce:
    python run_adaptive_baseline.py
"""

from __future__ import annotations
from artifact_io import output_options

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import qiskit
import qiskit_aer

from phase2_qaoa.adaptive_baseline import (
    ALPHA_KEEP_OLD,
    BATCH_SIZE,
    ELITE_FRACTION,
    aggregate_records,
    run_cem_seed,
)
from phase2_qaoa.qaoa_runner import evaluate_state as evaluate_geometric_state
from phase2_qaoa.qubo_encoder import enumerate_parameter_states
from phase3_quantum_tree.pipeline import evaluate_policy_state


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
SEEDS = list(range(42, 52))
K = 50


def _total_failures(evaluator) -> int:
    return int(
        sum(1 for state in enumerate_parameter_states() if evaluator(state)["failure"])
    )


def _run_variant(
    evaluator, evaluator_name: str, total_failures: int, out_name: str
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    for seed in SEEDS:
        record = run_cem_seed(
            seed=seed,
            evaluator=evaluator,
            evaluator_name=evaluator_name,
            total_failures=total_failures,
            k=K,
        )
        records.append(record)
        print(
            f"[{evaluator_name}] seed {seed}: "
            f"unique_failures={record['unique_failures']} "
            f"auc={record['auc']} "
            f"first_failure={record['time_to_first_failure']}"
        )

    aggregate = aggregate_records(records)
    seed42 = next(row for row in records if row["seed"] == 42)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "CEM adaptive baseline",
        "backend": {
            "classical_only": True,
            "quantum_layer_invoked": False,
            "qiskit_version": qiskit.__version__,
            "qiskit_aer_version": qiskit_aer.__version__,
            "note": "CEM is a purely classical adaptive search; no quantum backend is used.",
        },
        "config": {
            "seeds": SEEDS,
            "k": K,
            "sampling": "CEM adaptive without replacement",
            "evaluator": evaluator_name,
            "total_states": 256,
            "total_failures": total_failures,
            "batch_size": BATCH_SIZE,
            "elite_fraction": ELITE_FRACTION,
            "alpha_keep_old": ALPHA_KEEP_OLD,
            "hyperparameters_fixed_across_seeds": True,
        },
        "table_i_seed42_cem": {
            key: seed42[key]
            for key in (
                "seed",
                "k",
                "sampling",
                "unique_failures",
                "recall",
                "auc",
                "time_to_first_failure",
                "cumulative_failures",
            )
        },
        "records": records,
        "aggregate": aggregate,
    }
    out_path = OUT / out_name
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[{evaluator_name}] AGGREGATE unique_failures={aggregate['unique_failures']} "
        f"auc={aggregate['auc']} first_failure={aggregate['time_to_first_failure']}"
    )
    print(f"wrote {out_path}")
    return payload


def main() -> None:
    global OUT
    OUT = output_options(['adaptive_baseline.json', 'adaptive_baseline_policy.json'], "adaptive")
    OUT.mkdir(parents=True, exist_ok=True)
    geometric_total = _total_failures(evaluate_geometric_state)
    policy_total = _total_failures(evaluate_policy_state)
    if geometric_total != 18:
        raise RuntimeError(f"Expected 18 geometric failures, got {geometric_total}")
    if policy_total != 12:
        raise RuntimeError(f"Expected 12 policy failures, got {policy_total}")

    print("=== CEM adaptive baseline (geometric evaluator) ===")
    _run_variant(
        evaluate_geometric_state, "geometric", geometric_total, "adaptive_baseline.json"
    )
    print("")
    print("=== CEM adaptive baseline (policy evaluator) ===")
    _run_variant(
        evaluate_policy_state, "policy", policy_total, "adaptive_baseline_policy.json"
    )


if __name__ == "__main__":
    main()
