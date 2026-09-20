"""Positive-control and target-construction arms for the QTD attribution protocol.

Camera-ready addition (responds to the reviewers' request for a positive control).

Question: would the mechanism-control protocol of run_mechanism_controls.py detect
an informative Layer-2 score if one were present?  We substitute the circuit score
q_s of the teacher-gated top-25% with a *known informative* signal and check that
(i) the informative arm beats the full circuit and (ii) the shuffled-score control
"breaks", i.e. permuting the informative scores across gated candidates removes
the gain.  We also add two target-construction arms that isolate what the blend
itself does:

  zero_blend      q_s = 0 everywhere, blend kept (target = lambda(p_T) * p_T)
  constant        q_s = 0.5 for gated candidates (content-free gate indicator)
  oracle_a        q_s = a*label + (1-a)*U[0,1] for gated candidates, a in ORACLE_STRENGTHS
  oracle_a_shuf   the oracle_a scores permuted uniformly across gated candidates

All arms: in-sample geometric evaluator, seeds 42-51, k=50, pool 150, identical
warm start and lambda schedule, paired by seed against the persisted pipeline arms
(teacher_only, qtd_full), which are re-run here by the same code path.

Every arm is run twice and asserted identical before the JSON is written.
Output: outputs/positive_control.json  (never overwrites without --force)
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import qiskit

from phase3_quantum_tree.pipeline import run_quantum_tree_pipeline

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
SEEDS = list(range(42, 52))
K = 50
SHOTS = 1024
ORACLE_STRENGTHS = (1.0, 0.5, 0.25)
RNG_OFFSET = 10_007  # keeps control RNG streams disjoint from pipeline seeds


def auc(cumulative: Sequence[int]) -> float:
    previous, total = 0.0, 0.0
    for current in cumulative:
        total += (previous + float(current)) / 2.0
        previous = float(current)
    return round(total, 6)


def first_failure(cumulative: Sequence[int]) -> Optional[int]:
    return next((i for i, v in enumerate(cumulative, start=1) if int(v) > 0), None)


def mean_pstd(values: Sequence[float]) -> Dict[str, float]:
    return {"mean": round(float(statistics.mean(values)), 6), "std": round(float(statistics.pstdev(values)), 6)}


def bootstrap_mean_ci(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(0)
    samples = rng.choice(arr, size=(10_000, len(arr)), replace=True).mean(axis=1)
    return {
        "ci_95_low": round(float(np.quantile(samples, 0.025)), 6),
        "ci_95_high": round(float(np.quantile(samples, 0.975)), 6),
        "resamples": 10_000,
        "seed": 0,
    }


def sign_test(values: Sequence[float]) -> Dict[str, Any]:
    from math import comb

    pos = sum(1 for v in values if v > 0)
    neg = sum(1 for v in values if v < 0)
    n = pos + neg
    if n == 0:
        p = None
    else:
        tail = sum(comb(n, i) for i in range(0, min(pos, neg) + 1)) / 2**n
        p = round(min(1.0, 2.0 * tail), 6)
    return {"positive_count": pos, "negative_count": neg, "zero_count": len(values) - n, "sign_test_p": p}


# --------------------------------------------------------------------------- arms
def t_zero(scores, teacher_probs, labels, gated, seed):
    return np.zeros_like(scores)


def t_constant(scores, teacher_probs, labels, gated, seed):
    out = np.zeros_like(scores)
    out[gated] = 0.5
    return out


def make_oracle(strength: float, shuffled: bool) -> Callable[..., np.ndarray]:
    def transform(scores, teacher_probs, labels, gated, seed):
        rng = np.random.default_rng(seed + RNG_OFFSET)
        idx = np.flatnonzero(gated)
        noise = rng.uniform(0.0, 1.0, size=len(idx))
        values = strength * labels[idx] + (1.0 - strength) * noise
        if shuffled:
            values = np.random.default_rng(seed + 2 * RNG_OFFSET).permutation(values)
        out = np.zeros_like(scores)
        out[idx] = np.clip(values, 0.0, 1.0)
        return out

    return transform


def t_circuit_shuffled(scores, teacher_probs, labels, gated, seed):
    out = scores.copy()
    idx = np.flatnonzero(gated)
    out[idx] = np.random.default_rng(seed + 2 * RNG_OFFSET).permutation(scores[idx])
    return out


def arm_table() -> Dict[str, Dict[str, Any]]:
    arms: Dict[str, Dict[str, Any]] = {
        "teacher_only": {"mode": "teacher_only"},
        "qtd_full": {"mode": "qtd", "transform": None},
        "circuit_shuf": {"mode": "qtd", "transform": t_circuit_shuffled},
        "zero_blend": {"mode": "qtd", "transform": t_zero},
        "constant": {"mode": "qtd", "transform": t_constant},
    }
    for a in ORACLE_STRENGTHS:
        tag = f"oracle_{int(round(a * 100)):03d}"
        arms[tag] = {"mode": "qtd", "transform": make_oracle(a, shuffled=False)}
        arms[f"{tag}_shuf"] = {"mode": "qtd", "transform": make_oracle(a, shuffled=True)}
    return arms


def run_arm(seed: int, spec: Dict[str, Any], readout: str = "student") -> Dict[str, Any]:
    if spec["mode"] == "teacher_only":
        result = run_quantum_tree_pipeline(
            seed=seed, shots=SHOTS, k_iterations=K, quantum_enabled=False,
            ranking_mode="teacher_only", evaluator="geometric", output_path=None, readout=readout,
        )
    else:
        result = run_quantum_tree_pipeline(
            seed=seed, shots=SHOTS, k_iterations=K, quantum_enabled=True, quantum_variant="full",
            ranking_mode="qtd", evaluator="geometric", output_path=None,
            score_transform=spec.get("transform"), readout=readout,
        )
    cumulative = [int(v) for v in result["cumulative_failures"]]
    return {
        "seed": seed,
        "resources": result["resources"],
        "unique_failures": int(result["total_unique_failures"]),
        "auc": auc(cumulative),
        "first_failure": first_failure(cumulative),
        "cumulative_failures": cumulative,
    }


def paired(rows: Dict[str, List[Dict[str, Any]]], arm: str, ref: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"reference": ref}
    for metric in ("auc", "unique_failures"):
        deltas = [float(a[metric]) - float(b[metric]) for a, b in zip(rows[arm], rows[ref])]
        out[f"{metric}_delta"] = {
            "per_seed": deltas,
            "aggregate": mean_pstd(deltas),
            "bootstrap_mean_95_ci": bootstrap_mean_ci(deltas),
            "exact_tests": sign_test(deltas),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-name", default="positive_control.json")
    args = parser.parse_args()
    out_path = OUT / args.output_name
    if out_path.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {out_path}; pass --force")

    arms = arm_table()
    results: Dict[str, Any] = {}
    determinism: Dict[str, Any] = {}
    for readout in ("student", "target"):
        rows: Dict[str, List[Dict[str, Any]]] = {}
        determinism[readout] = {}
        print(f"--- readout = {readout}")
        for name, spec in arms.items():
            rows[name], determinism[readout][name] = [], {}
            for seed in SEEDS:
                first = run_arm(seed, spec, readout)
                second = run_arm(seed, spec, readout)
                if first != second:
                    raise RuntimeError(f"non-deterministic arm {name} seed {seed}")
                determinism[readout][name][str(seed)] = True
                rows[name].append(first)
            agg = {m: mean_pstd([float(r[m]) for r in rows[name]]) for m in ("unique_failures", "auc")}
            print(f"{name:18s} failures {agg['unique_failures']}  auc {agg['auc']}")

        comparisons: Dict[str, Any] = {}
        for name in arms:
            comparisons[name] = {
                "vs_teacher_only": paired(rows, name, "teacher_only"),
                "vs_qtd_full": paired(rows, name, "qtd_full"),
            }
            if name.startswith("oracle_") and not name.endswith("_shuf"):
                comparisons[name]["vs_own_shuffle"] = paired(rows, name, f"{name}_shuf")
        comparisons["qtd_full"]["vs_own_shuffle"] = paired(rows, "qtd_full", "circuit_shuf")
        comparisons["constant"]["vs_zero_blend"] = paired(rows, "constant", "zero_blend")
        comparisons["qtd_full"]["vs_zero_blend"] = paired(rows, "qtd_full", "zero_blend")
        comparisons["qtd_full"]["vs_constant"] = paired(rows, "qtd_full", "constant")
        results[readout] = {
            "arms": {
                name: {
                    "per_seed": rows[name],
                    "aggregate": {m: mean_pstd([float(r[m]) for r in rows[name] if r[m] is not None]) for m in ("unique_failures", "auc", "first_failure")},
                }
                for name in arms
            },
            "paired_comparisons": comparisons,
        }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Positive control and target-construction arms for the QTD attribution protocol",
        "config": {
            "seeds": SEEDS, "k": K, "pool": 3 * K, "shots": SHOTS, "evaluator": "geometric",
            "oracle_strengths": list(ORACLE_STRENGTHS),
            "oracle_definition": "q_s = a*label + (1-a)*U[0,1] on teacher-gated top-25%; 0 elsewhere",
            "rng": {"noise": "default_rng(seed + 10007)", "shuffle": "default_rng(seed + 20014)"},
        },
        "backend": {"qiskit_version": qiskit.__version__},
        "determinism_checked": determinism,
        "readouts": results,
        "readout_definition": {"student": "pool ranked by the logistic student (the paper pipeline)", "target": "pool ranked directly by the distillation target (bypasses the student)"},
        "status": "post hoc; added for the camera-ready in response to reviewer request",
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
