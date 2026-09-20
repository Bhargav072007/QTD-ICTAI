"""QUBO diagnostics added for the camera-ready (reviewer requests on RQ1).

1. Ranking AUC of the rollout-derived QUBO objective against the 18 geometric
   failures, with and without the fixed "11" penalty, under both the symmetric
   x^T Q x convention and the upper-triangular (Hamiltonian-energy) convention.
2. Per-parameter marginal ranking AUC (each state scored by the mean objective of
   its value of that parameter).
3. Exact-enumeration comparator: all 256 states sorted by objective; failures
   among the N lowest-energy states.  This is a deterministic energy-ranking baseline, not a ceiling on
   the failure discovery of arbitrary optimizers or sampling methods.

Deterministic; classical only.  Output: outputs/qubo_diagnostics.json
Run from the repository root:  python analysis/qubo_diagnostics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from artifact_io import output_options

from phase2_qaoa import qubo_encoder as Q  # noqa: E402
from phase2_qaoa.qaoa_runner import evaluate_state  # noqa: E402

NAMES = ["heading", "altitude", "speed", "lateral_offset"]


def rank_auc(score: np.ndarray, labels: np.ndarray) -> float:
    score = np.round(score, 12)  # Numerical noise must not split mathematical ties.
    pos, neg = score[labels == 1], score[labels == 0]
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def main() -> None:
    destination = output_options(["qubo_diagnostics.json"], "derived")
    states = Q.enumerate_parameter_states()
    labels = np.array([1 if evaluate_state(s)["failure"] else 0 for s in states])
    idx = np.array([Q._state_indices_from_params(s) for s in states])
    X = np.array([[int(c) for c in Q.encode_state_indices(tuple(i))[::-1]] for i in idx], dtype=float)

    q_pen = Q.build_qubo_matrix()
    original = Q._apply_one_hot_regularization
    Q._apply_one_hot_regularization = lambda qubo: None
    try:
        q_nopen = Q.build_qubo_matrix()
    finally:
        Q._apply_one_hot_regularization = original

    out = {"state_count": int(len(states)), "geometric_failures": int(labels.sum()), "variants": {}, "implemented_convention": "upper_triangular_hamiltonian",
           "tie_break": "round energies to 12 decimals, then stable enumerate_parameter_states order",
           "energy_ranking_decimals": 12,
           "resource_accounting": {"diagnostic_label_calls": len(states), "unique_states": len(states), "circuit_shots": 0}}
    for tag, mat in (("with_penalty", q_pen), ("without_penalty", q_nopen)):
        entry = {}
        for conv, m in (("xTQx_symmetric", mat), ("upper_triangular_hamiltonian", np.triu(mat))):
            obj = (np.array([Q.objective_value(mat, x) for x in X]) if conv == "upper_triangular_hamiltonian"
                   else np.einsum("ni,ij,nj->n", X, m, X))
            obj = np.round(obj, 12)
            order = np.argsort(obj, kind="stable")
            cum = np.cumsum(labels[order])
            best = states[int(order[0])]
            per_param = {}
            for p, name in enumerate(NAMES):
                means = [float(obj[idx[:, p] == v].mean()) for v in range(4)]
                marginal = -np.array([means[v] for v in idx[:, p]])
                per_param[name] = {
                    "mean_objective_by_value_index": [round(x, 6) for x in means],
                    "failures_by_value_index": [int(labels[idx[:, p] == v].sum()) for v in range(4)],
                    "marginal_ranking_auc": round(rank_auc(marginal, labels), 6),
                }
            entry[conv] = {
                "ranking_auc": round(rank_auc(-obj, labels), 6),
                "failures_in_lowest_energy": {str(n): int(cum[n - 1]) for n in range(1, len(states) + 1)},
                "rank_of_first_failure": int(np.argmax(labels[order] == 1)) + 1,
                "global_minimum_state": best,
                "global_minimum_is_failure": bool(labels[order[0]]),
                "exact_enumeration_cumulative_failures_k50": [int(v) for v in cum[:50]],
                "per_parameter": per_param,
            }
        out["variants"][tag] = entry
    out["note"] = (
        "Value index 0 of each parameter encodes as bits 00; the frequency-based construction credits only "
        "set bits, so index 0 is never rewarded.  12 of 18 failures have heading index 0."
    )
    corrected_path = destination / "qaoa_multiseed.json"
    if corrected_path.exists():
        corrected = json.loads(corrected_path.read_text())
        prefix = out["variants"]["with_penalty"]["upper_triangular_hamiltonian"]["failures_in_lowest_energy"]
        records = [{"seed": row["seed"], "unique_state_budget": row["unique_env_evaluations"],
                    "qaoa_failures": row["unique_failures"],
                    "energy_ranking_failures": prefix[str(row["unique_env_evaluations"]) ]}
                   for row in corrected["per_seed"]]
        values = np.array([row["energy_ranking_failures"] for row in records])
        out["qaoa_matched_unique_budgets"] = {
            "source": str(corrected_path), "per_seed": records,
            "energy_ranking_mean": float(values.mean()), "energy_ranking_population_sd": float(values.std()),
            "note": "Variation reflects QAOA's changing state budget; the energy ranking itself is deterministic. Not a universal ceiling.",
        }
    path = destination / "qubo_diagnostics.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    sym = out["variants"]["with_penalty"]["xTQx_symmetric"]
    print("AUC with penalty", sym["ranking_auc"], "| without", out["variants"]["without_penalty"]["xTQx_symmetric"]["ranking_auc"])
    print("lowest-energy failures", sym["failures_in_lowest_energy"], "| min is failure:", sym["global_minimum_is_failure"])
    print({k: v["marginal_ranking_auc"] for k, v in sym["per_parameter"].items()})
    print("wrote", path)


if __name__ == "__main__":
    main()
