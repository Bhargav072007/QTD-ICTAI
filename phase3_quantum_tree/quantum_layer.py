"""
Layer 2: quantum refinement/distillation over classical teacher outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit.primitives import StatevectorSampler
    from qiskit.quantum_info import Statevector

    QISKIT_TREE_AVAILABLE = True
except ImportError:
    QuantumCircuit = None
    StatevectorSampler = None
    Statevector = None
    QISKIT_TREE_AVAILABLE = False


@dataclass
class QuantumDistillationLayer:
    """
    Variational quantum encoding layer that refines teacher confidence scores.

    IMPORTANT: This layer is NOT QAOA (Quantum Approximate Optimization Algorithm).
    QAOA requires a cost Hamiltonian, mixing unitary, and classical outer-loop
    parameter optimization. This layer instead encodes teacher hidden activations
    as RY rotation angles on a 3-qubit circuit, applies CZ entanglement, and
    measures the average Hamming weight of the output distribution.

    Architecture per candidate:
      - 3 qubits
      - RY(theta_i) for each qubit, where theta_i = clip(|h_i| / max|h|, 0, 1) * pi
      - CZ(0,1), CZ(1,2) entanglement
      - Final RY(teacher_prob * pi) on qubit 0
      - quantum_score = mean Hamming weight of measurement outcomes

    Only the top-25% of candidates (by teacher confidence) are processed;
    the remainder receive quantum_score=0 (skipped-low-teacher).
    """

    shots: int = 256
    seed: int = 42
    variant: str = "full"

    def _top_indices(self, teacher_probs: np.ndarray) -> np.ndarray:
        top_k = max(1, int(np.ceil(len(teacher_probs) * 0.25)))
        return np.argsort(teacher_probs)[::-1][:top_k]

    @staticmethod
    def _angles(row: np.ndarray, teacher_prob: float) -> tuple[np.ndarray, float]:
        features = np.abs(row[:3])
        scale = np.max(features) if np.max(features) > 0 else 1.0
        theta = np.clip(features / scale, 0.0, 1.0) * np.pi
        phi = float(np.clip(teacher_prob, 0.0, 1.0) * np.pi)
        return theta, phi

    def _matched_classical_refine(self, hidden: np.ndarray, teacher_probs: np.ndarray) -> Dict[str, np.ndarray]:
        """Exact no-CZ marginal formula for the implemented product circuit.

        The final teacher RY acts on qubit zero, so its no-CZ marginal is
        sin^2((theta_0 + phi)/2), not an independently averaged probability.
        """
        top_indices = self._top_indices(teacher_probs)
        scores = np.zeros(len(hidden), dtype=float)
        backend = np.array(["skipped-low-teacher"] * len(hidden), dtype=object)
        for index in top_indices:
            theta, phi = self._angles(hidden[int(index)], float(teacher_probs[int(index)]))
            marginals = np.array([
                np.sin((theta[0] + phi) / 2.0) ** 2,
                np.sin(theta[1] / 2.0) ** 2,
                np.sin(theta[2] / 2.0) ** 2,
            ])
            scores[int(index)] = float(np.mean(marginals))
            backend[int(index)] = "matched-classical-no-cz"
        return {"backend": backend, "quantum_scores": scores}

    def _fallback_refine(self, hidden: np.ndarray, teacher_probs: np.ndarray) -> Dict[str, np.ndarray]:
        top_k = max(1, int(np.ceil(len(hidden) * 0.25)))
        top_indices = np.argsort(teacher_probs)[::-1][:top_k]
        hidden_energy = np.mean(np.square(hidden), axis=1)
        hidden_energy = hidden_energy / max(np.max(hidden_energy), 1e-8)
        refined = np.zeros(len(hidden), dtype=float)
        refined[top_indices] = np.clip(
            0.58 * teacher_probs[top_indices] + 0.42 * hidden_energy[top_indices],
            0.0,
            1.0,
        )
        return {
            "backend": np.array(
                ["classical-fallback" if index in set(top_indices.tolist()) else "skipped-low-teacher" for index in range(len(hidden))],
                dtype=object,
            ),
            "quantum_scores": refined,
        }

    def _qiskit_refine(self, hidden: np.ndarray, teacher_probs: np.ndarray) -> Dict[str, np.ndarray]:
        sampler = StatevectorSampler(seed=self.seed)
        top_indices = self._top_indices(teacher_probs)
        top_index_set = set(int(