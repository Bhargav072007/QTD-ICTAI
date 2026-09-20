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

    def _build_circuit(self, theta: np.ndarray, phi: float, entangle: bool = True, measure: bool = True):
        qc = QuantumCircuit(3)
        for qubit in range(3):
            qc.ry(float(theta[qubit]), qubit)
        if entangle:
            qc.cz(0, 1)
            qc.cz(1, 2)
        qc.ry(phi, 0)
        if measure:
            qc.measure_all()
        return qc

    def _qiskit_refine(self, hidden: np.ndarray, teacher_probs: np.ndarray, entangle: bool = True) -> Dict[str, np.ndarray]:
        """Shot-sampled 3-qubit circuit (variant "full"; entangle=False gives "no_cz").

        NOTE (camera-ready repair): the body of this method was truncated in the
        first public export. It was reconstructed from the class docstring and
        validated against the persisted per-candidate scores in
        outputs/ranking_diagnostic_seed42.json and the per-seed curves in
        outputs/mechanism_controls.json (see outputs/repair_validation.json).
        """
        sampler = StatevectorSampler(seed=self.seed)
        top_indices = self._top_indices(teacher_probs)
        top_index_set = set(int(index) for index in top_indices.tolist())
        scores = np.zeros(len(hidden), dtype=float)
        backend = np.array(["skipped-low-teacher"] * len(hidden), dtype=object)
        for index in range(len(hidden)):
            if index not in top_index_set:
                continue
            theta, phi = self._angles(hidden[index], float(teacher_probs[index]))
            qc = self._build_circuit(theta, phi, entangle=entangle, measure=True)
            result = sampler.run([qc], shots=int(self.shots)).result()[0]
            counts = result.data.meas.get_counts()
            total = sum(counts.values())
            weight = sum(bits.count("1") * count for bits, count in counts.items())
            scores[index] = float(np.clip(weight / (3.0 * total), 0.0, 1.0))
            backend[index] = "qiskit-statevector" if entangle else "qiskit-statevector-no-cz"
        return {"backend": backend, "quantum_scores": scores}

    def _analytic_refine(self, hidden: np.ndarray, teacher_probs: np.ndarray) -> Dict[str, np.ndarray]:
        """Exact (shot-free) expectation of the same entangled circuit."""
        top_indices = self._top_indices(teacher_probs)
        scores = np.zeros(len(hidden), dtype=float)
        backend = np.array(["skipped-low-teacher"] * len(hidden), dtype=object)
        for index in top_indices:
            theta, phi = self._angles(hidden[int(index)], float(teacher_probs[int(index)]))
            qc = self._build_circuit(theta, phi, entangle=True, measure=False)
            probabilities = Statevector.from_instruction(qc).probabilities_dict()
            scores[int(index)] = float(
                np.clip(sum((bits.count("1") / 3.0) * float(p) for bits, p in probabilities.items()), 0.0, 1.0)
            )
            backend[int(index)] = "qiskit-statevector-exact"
        return {"backend": backend, "quantum_scores": scores}

    def refine(self, hidden: np.ndarray, teacher_probs: np.ndarray) -> Dict[str, np.ndarray]:
        hidden = np.asarray(hidden, dtype=float)
        teacher_probs = np.asarray(teacher_probs, dtype=float).reshape(-1)
        if self.variant == "matched_classical":
            return self._matched_classical_refine(hidden, teacher_probs)
        if not QISKIT_TREE_AVAILABLE:
            return self._fallback_refine(hidden, teacher_probs)
        if self.variant == "full":
            return self._qiskit_refine(hidden, teacher_probs, entangle=True)
        if self.variant == "no_cz":
            return self._qiskit_refine(hidden, teacher_probs, entangle=False)
        if self.variant == "analytic":
            return self._analytic_refine(hidden, teacher_probs)
        if self.variant in {"shuffled", "random"}:
            result = self._qiskit_refine(hidden, teacher_probs, entangle=True)
            top_indices = np.sort(self._top_indices(teacher_probs))
            rng = np.random.default_rng(self.seed)
            scores = result["quantum_scores"].copy()
            if self.variant == "shuffled":
                scores[top_indices] = rng.permutation(scores[top_indices])
            else:
                scores[top_indices] = rng.uniform(0.0, 1.0, size=len(top_indices))
            result["quantum_scores"] = scores
            return result
        raise ValueError(f"unknown quantum variant: {self.variant}")
