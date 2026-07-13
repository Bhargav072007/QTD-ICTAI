#!/usr/bin/env python3
"""Canonical QUBO/failure-label alignment audit.

This script intentionally refuses to overwrite its output. It compares the
current empirical QUBO against the legacy hand-shaped surrogate QUBO over all
256 grid states.
"""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from scipy.stats import pointbiserialr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase2_qaoa.qaoa_runner import evaluate_state  # noqa: E402
from phase2_qaoa.qubo_encoder import (  # noqa: E402
    PARAM_GRID,
    PARAM_ORDER,
    _build_surrogate_qubo_matrix,
    build_qubo_matrix,
    encode_state_indices,
    enumerate_parameter_states,
    load_empirical_failure_probabilities,
    qubo_to_hamiltonian,
)

DEFAULT_OUTPUT_PATH = ROOT / "outputs" / "qubo_alignment.json"


def state_indices(state: Dict[str, float]) -> Tuple[int, int, int, int]:
    return tuple(PARAM_GRID[name].index(float(state[name])) for name in PARAM_ORDER)


def encoded_bitstring(state: Dict[str, float]) -> str:
    return encode_state_indices(state_indices(state))


def qubo_vector_from_state(state: Dict[str, float]) -> np.ndarray:
    # build_qubo_matrix() uses encode_state_indices(indices)[::-1] as QUBO bit order.
    return np.array([int(bit) for bit in encoded_bitstring(state)[::-1]], dtype=float)


def objective_values(qubo: np.ndarray, states: Iterable[Dict[str, float]]) -> np.ndarray:
    values: List[float] = []
    for state in states:
        x = qubo_vector_from_state(state)
        values.append(float(x @ qubo @ x))
    return np.array(values, dtype=float)


def upper_triangular_objective_values(qubo: np.ndarray, states: Iterable[Dict[str, float]]) -> np.ndarray:
    values: List[float] = []
    diag = np.diag(qubo)
    for state in states:
        x = qubo_vector_from_state(state)
        value = float(np.sum(diag * x))
        for i in range(qubo.shape[0]):
            for j in range(i + 1, qubo.shape[1]):
                value += float(qubo[i, j] * x[i] * x[j])
        values.append(value)
    return np.array(values, dtype=float)


def hamiltonian_diagonal_values(qubo: np.ndarray, states: Iterable[Dict[str, float]]) -> np.ndarray:
    hamiltonian = qubo_to_hamiltonian(qubo)
    values: List[float] = []
    for state in states:
        x = qubo_vector_from_state(state)
        z_values = [1 - 2 * int(bit) for bit in x.astype(int)]
        total = 0.0
        for label, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
            term = 1.0
            for index, pauli in enumerate(reversed(str(label))):
                if pauli == "Z":
                    term *= z_values[index]
            total += float(complex(coeff).real) * term
        values.append(total)
    return np.array(values, dtype=float)


def metric_row(
    labels: np.ndarray,
    objective: np.ndarray,
    upper_objective: np.ndarray,
    hamiltonian_energy: np.ndarray,
) -> Dict[str, Any]:
    score = -objective
    # Rounding avoids meaningless ROC tie-break changes from ~1e-15 Pauli-sum noise.
    upper_objective_for_auc = np.round(upper_objective, 12)
    hamiltonian_energy_for_auc = np.round(hamiltonian_energy, 12)
    upper_score = -upper_objective_for_auc
    hamiltonian_score = -hamiltonian_energy_for_auc
    pb = pointbiserialr(labels.astype(int), objective)
    upper_pb = pointbiserialr(labels.astype(int), upper_objective)
    hamiltonian_pb = pointbiserialr(labels.astype(int), hamiltonian_energy)
    auc = float(roc_auc_score(labels.astype(int), score))
    upper_auc = float(roc_auc_score(labels.astype(int), upper_score))
    hamiltonian_auc = float(roc_auc_score(labels.astype(int), hamiltonian_score))
    hamiltonian_delta = hamiltonian_energy - upper_objective
    hamiltonian_centered_delta = hamiltonian_delta - float(np.mean(hamiltonian_delta))
    return {
        "positive_labels": int(np.sum(labels)),
        "negative_labels": int(len(labels) - np.sum(labels)),
        "point_biserial_r_label_vs_objective": float(pb.statistic),
        "point_biserial_pvalue": float(pb.pvalue),
        "ranking_auc_score_neg_objective": auc,
        "upper_triangular_point_biserial_r_label_vs_objective": float(upper_pb.statistic),
        "upper_triangular_point_biserial_pvalue": float(upper_pb.pvalue),
        "upper_triangular_ranking_auc_score_neg_objective": upper_auc,
        "hamiltonian_point_biserial_r_label_vs_energy": float(hamiltonian_pb.statistic),
        "hamiltonian_point_biserial_pvalue": float(hamiltonian_pb.pvalue),
        "hamiltonian_ranking_auc_score_neg_energy": hamiltonian_auc,
        "hamiltonian_auc_matches_double_counted_xtqx_auc": bool(np.isclose(auc, hamiltonian_auc, atol=1e-12)),
        "hamiltonian_auc_matches_upper_triangular_auc": bool(np.isclose(upper_auc, hamiltonian_auc, atol=1e-12)),
        "hamiltonian_auc_rounding_decimals": 12,
        "hamiltonian_minus_upper_triangular_objective_constant": float(np.mean(hamiltonian_delta)),
        "hamiltonian_minus_upper_triangular_objective_centered_max_abs": float(
            np.max(np.abs(hamiltonian_centered_delta))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute QUBO/failure-label alignment metrics.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    output_path = args.output if args.output.is_absolute() else ROOT / args.output

    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    states = enumerate_parameter_states()
    empirical_probabilities = load_empirical_failure_probabilities()
    policy_failure_bitstrings = set(empirical_probabilities.keys())

    geometric_labels = np.array([bool(evaluate_state(state)["failure"]) for state in states], dtype=bool)
    policy_labels = np.array([encoded_bitstring(state) in policy_failure_bitstrings for state in states], dtype=bool)

    qubos = {
        "empirical_build_qubo_matrix": build_qubo_matrix(),
        "surrogate_fallback": _build_surrogate_qubo_matrix(),
    }

    results: Dict[str, Any] = {
        "state_count": len(states),
        "geometric_failure_count": int(np.sum(geometric_labels)),
        "policy_failure_count_from_empirical_probabilities": int(np.sum(policy_labels)),
        "empirical_probability_count": len(empirical_probabilities),
        "bit_convention": "x = encode_state_indices(indices)[::-1] as a 0/1 vector",
        "qaoa_minimization_score": "primary score = -(x^T Q x)",
        "hamiltonian_note": (
            "qubo_to_hamiltonian(Q) matches the single-count upper-triangular QUBO energy; "
            "for symmetric Q matrices, x^T Q x double-counts off-diagonal terms."
        ),
        "qubo_sources": {
            "empirical_build_qubo_matrix": "build_qubo_matrix(), empirical when load_empirical_failure_probabilities() is non-empty",
            "surrogate_fallback": "_build_surrogate_qubo_matrix()",
        },
        "metrics": {},
    }

    for qubo_name, qubo in qubos.items():
        objective = objective_values(qubo, states)
        upper_objective = upper_triangular_objective_values(qubo, states)
        hamiltonian_energy = hamiltonian_diagonal_values(qubo, states)
        results["metrics"][qubo_name] = {
            "geometric_labels": metric_row(geometric_labels, objective, upper_objective, hamiltonian_energy),
            "policy_labels": metric_row(policy_labels, objective, upper_objective, hamiltonian_energy),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {output_path.relative_to(ROOT)}")
    print(f"state_count={results['state_count']}")
    print(f"geometric_failure_count={results['geometric_failure_count']}")
    print(f"policy_failure_count={results['policy_failure_count_from_empirical_probabilities']}")
    print(f"empirical_probability_count={results['empirical_probability_count']}")
    print("qubo_type,label_set,xtqx_r,xtqx_auc,hamiltonian_auc,xtqx_hamiltonian_auc_match,upper_hamiltonian_auc_match")
    for qubo_name, by_label in results["metrics"].items():
        for label_name, row in by_label.items():
            print(
                ",".join(
                    [
                        qubo_name,
                        label_name,
                        f"{row['point_biserial_r_label_vs_objective']:.12f}",
                        f"{row['ranking_auc_score_neg_objective']:.12f}",
                        f"{row['hamiltonian_ranking_auc_score_neg_energy']:.12f}",
                        str(row["hamiltonian_auc_matches_double_counted_xtqx_auc"]),
                        str(row["hamiltonian_auc_matches_upper_triangular_auc"]),
                    ]
                )
            )


if __name__ == "__main__":
    main()
