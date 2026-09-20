"""
QAA Phase 2 - QAOA-style exploration runner.

Paper parameters: --reps 2 --shots 1024 --k 50 --seed 42
Classical optimizer: COBYLA with maxiter=35 per iteration.

Paper-replicating invocation:
    python phase2_qaoa/qaoa_runner.py --reps 2 --shots 1024 --k 50 --seed 42
Fast portal invocation (not paper-comparable):
    python phase2_qaoa/qaoa_runner.py --reps 1 --shots 256 --k 10 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from qiskit.circuit.library import qaoa_ansatz
    from qiskit.primitives import StatevectorSampler
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_algorithms.utils import algorithm_globals

    QISKIT_AVAILABLE = True
except ImportError:
    qaoa_ansatz = None
    StatevectorSampler = None
    COBYLA = None
    algorithm_globals = None
    QISKIT_AVAILABLE = False

from phase2_qaoa.qubo_encoder import (
    N_QUBITS,
    PARAM_GRID,
    decode_bitstring,
    encode_state_indices,
    enumerate_parameter_states,
    estimate_failure_score,
    get_hamiltonian,
)

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def _state_key(params: Dict[str, float]) -> str:
    return "|".join(f"{name}={params[name]}" for name in sorted(params))


def _failure_type(result: Dict[str, Any]) -> str:
    if result["separation_loss"]:
        return "separation_loss"
    if result["near_miss"]:
        return "near_miss"
    return "safe"


def evaluate_state(params: Dict[str, float]) -> Dict[str, Any]:
    """
    Lightweight aviation encounter model aligned with the paper scope.
    """
    # SIMPLIFIED MODEL: ego altitude is fixed at 31000 ft throughout.
    # phase1/aviation_env_3d.py apply ego_climb_rate per step when action != "maintain".
    # Results from this function are NOT directly comparable to phase1 evaluation.
    # Both use horizon=60 steps and the same separation thresholds.
    ego_x, ego_y, ego_z = 20.0, 100.0, 31000.0
    ego_heading_deg, ego_speed = 0.0, 4.5
    intruder_x = 88.0 + params["int_x_offset"]
    intruder_y = 10.0
    intruder_z = params["int_altitude"]
    intruder_heading_deg = params["int_heading"]
    intruder_speed = params["int_speed"]

    ex, ey = ego_x, ego_y
    ix, iy = intruder_x, intruder_y
    min_h_sep = float("inf")
    min_v_sep = float("inf")
    separation_loss = False
    near_miss = False

    for _ in range(60):
        ex += ego_speed * math.cos(math.radians(ego_heading_deg))
        ey += ego_speed * math.sin(math.radians(ego_heading_deg))
        ix += intruder_speed * math.cos(math.radians(intruder_heading_deg))
        iy += intruder_speed * math.sin(math.radians(intruder_heading_deg))

        h_sep = math.hypot(ex - ix, ey - iy)
        v_sep = abs(ego_z - intruder_z)
        min_h_sep = min(min_h_sep, h_sep)
        min_v_sep = min(min_v_sep, v_sep)

        if h_sep < 5.0 and v_sep < 1000.0:
            separation_loss = True
        if h_sep < 7.5 and v_sep < 1500.0:
            near_miss = True

    result = {
        "params": params,
        "min_h_sep_nm": round(min_h_sep, 3),
        "min_v_sep_ft": round(min_v_sep, 1),
        "separation_loss": separation_loss,
        "near_miss": near_miss,
        "failure": separation_loss or near_miss,
    }
    result["failure_type"] = _failure_type(result)
    result["surrogate_score"] = round(estimate_failure_score(params), 4)
    return result


@dataclass
class FallbackCircuit:
    num_qubits: int
    reps: int

    @property
    def num_parameters(self) -> int:
        return self.reps * 2

    def depth(self) -> int:
        return 6 * self.reps + 10

    def assign_parameters(self, params: np.ndarray) -> np.ndarray:
        return np.array(params, dtype=float)

    def draw(self, output: str = "text", fold: int = 80) -> str:
        del output, fold
        return (
            "     [H]   cost(ZZ)   mix(RX)\n"
            "q_0: [H]---*----------*------\n"
            "q_1: [H]---*----------*------\n"
            "q_2: [H]---*----------*------\n"
            "q_3: [H]---*----------*------\n"
            "q_4: [H]---*----------*------\n"
            "q_5: [H]---*----------*------\n"
            "q_6: [H]---*----------*------\n"
            "q_7: [H]---*----------*------"
        )


class QAOAExplorer:
    """
    Runs a Qiskit-first QAOA exploration loop with a deterministic local fallback.

    Classical optimizer: COBYLA with maxiter=35 per iteration.
    """

    def __init__(self, reps: int = 2, shots: int = 1024, seed: int = 42) -> None:
        self.reps = reps
        # shots=1024 is the paper's reported value.
        # The portal benchmark (app.py:run_assurance_benchmark) uses shots=256, k=10
        # for speed. For paper-replicating results, use: --shots 1024 --k 50
        self.shots = shots
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.hamiltonian, self.qubo = get_hamiltonian()
        self.all_failures: List[Dict[str, Any]] = []
        self.iteration_counts: List[int] = []
        self._known_keys: set[str] = set()
        # Environment-evaluation accounting for the exploration loop. run_iteration()
        # evaluates the top-8 bitstrings each iteration, so these count how many times
        # evaluate_state() is called during exploration (total, non-unique) and how many
        # distinct states that touches. They exclude the one-off catalog build below.
        self.n_env_evaluations_total = 0
        self._env_eval_unique_keys: set[str] = set()
        # The Qiskit path never consumes the fallback's oracle-labeled catalog.
        self._evaluated_states = [] if QISKIT_AVAILABLE else self._build_state_catalog()
        self.optimizer_evaluations = 0
        self.optimizer_shots = 0
        self.sampling_calls = 0
        self.sampling_shots = 0
        self.optimizer_nfev = []

        if QISKIT_AVAILABLE:
            algorithm_globals.random_seed = seed
            self.circuit = qaoa_ansatz(
                cost_operator=self.hamiltonian,
                reps=reps,
                insert_barriers=True,
                flatten=True,
            )
            self.sampler = StatevectorSampler(seed=seed)
            self.backend_name = "Qiskit StatevectorSampler"
        else:
            self.circuit = FallbackCircuit(num_qubits=N_QUBITS, reps=reps)
            self.sampler = None
            self.backend_name = "Deterministic fallback sampler"

    def _build_state_catalog(self) -> List[Dict[str, Any]]:
        catalog: List[Dict[str, Any]] = []
        for state in enumerate_parameter_states():
            heading_index = PARAM_GRID["int_heading"].index(state["int_heading"])
            altitude_index = PARAM_GRID["int_altitude"].index(state["int_altitude"])
            speed_index = PARAM_GRID["int_speed"].index(state["int_speed"])
            offset_index = PARAM_GRID["int_x_offset"].index(state["int_x_offset"])
            bitstring = encode_state_indices(
                [heading_index, altitude_index, speed_index, offset_index]
            )
            evaluation = evaluate_state(state)
            evaluation["bitstring"] = bitstring
            catalog.append(evaluation)
        return catalog

    @lru_cache(maxsize=256)
    def _hamiltonian_cost(self, bitstring: str) -> float:
        # Qiskit strings display q[n-1] ... q[0]; reversed Pauli labels
        # below enumerate q[0] first. Preserve term order and arithmetic.
        if len(bitstring) != N_QUBITS or set(bitstring) - {"0", "1"}:
            raise ValueError("Expected an 8-bit Qiskit measurement string")
        z_values = [1 - 2 * int(bit) for bit in reversed(bitstring)]
        total = 0.0
        for label, coeff in zip(self.hamiltonian.paulis, self.hamiltonian.coeffs):
            term = 1.0
            for index, pauli in enumerate(reversed(str(label))):
                if pauli == "Z":
                    term *= z_values[index]
            total += float(coeff.real) * term
        return total

    def _fallback_counts(self, params: np.ndarray) -> Dict[str, int]:
        logits: List[float] = []
        for index, state in enumerate(self._evaluated_states):
            jitter = 0.03 * math.sin(float(np.sum(params)) + index)
            score = (
                1.6 * state["surrogate_score"]
                + 0.7 * float(state["failure"])
                + jitter
                - 0.15 * self._hamiltonian_cost(state["bitstring"])
            )
            logits.append(score)
        centered = np.array(logits) - max(logits)
        probs = np.exp(centered)
        probs = probs / probs.sum()
        sampled = self.rng.multinomial(self.shots, probs)
        return {
            state["bitstring"]: int(count)
            for state, count in zip(self._evaluated_states, sampled)
            if count > 0
        }

    def _qiskit_counts(self, params: np.ndarray) -> Dict[str, int]:
        bound = self.circuit.assign_parameters(params)
        measured = bound.measure_all(inplace=False)
        result = self.sampler.run([measured], shots=self.shots).result()
        return result[0].data.meas.get_counts()

    def _sample_counts(self, params: np.ndarray) -> Dict[str, int]:
        if QISKIT_AVAILABLE:
            return self._qiskit_counts(params)
        return self._fallback_counts(params)

    def _evaluate_circuit(self, params: np.ndarray) -> float:
        counts = self._sample_counts(params)
        self.optimizer_evaluations += 1
        self.optimizer_shots += sum(counts.values())
        total_shots = max(sum(counts.values()), 1)
        expected_cost = 0.0
        for bitstring, count in counts.items():
            expected_cost += (count / total_shots) * self._hamiltonian_cost(bitstring)
        return expected_cost

    def _optimize_params(self, iteration: int) -> np.ndarray:
        if QISKIT_AVAILABLE and COBYLA is not None:
            init = self.rng.uniform(0, 2 * math.pi, self.circuit.num_parameters)
            # COBYLA maxiter=35 bounds the classical outer-loop optimizer to at most 35
            # cost-function (circuit-expectation) evaluations per QAOA iteration. This
            # optimizer budget is separate from environment evaluations: the exploration
            # loop additionally evaluates the top-8 bitstrings each iteration, so over k
            # iterations it performs up to 8*k non-unique environment evaluations
            # (tracked as n_env_evaluations_total / n_env_evaluations_unique in run()).
            before = self.optimizer_evaluations
            result = COBYLA(maxiter=35).minimize(fun=self._evaluate_circuit, x0=init)
            measured = self.optimizer_evaluations - before
            if measured != result.nfev:
                raise RuntimeError("Optimizer evaluation accounting disagrees with nfev")
            self.optimizer_nfev.append(measured)
            return np.array(result.x, dtype=float)

        candidates = []
        for offset in range(10):
            local_rng = np.random.default_rng(self.seed + iteration * 17 + offset)
            candidate = local_rng.uniform(0, 2 * math.pi, self.circuit.num_parameters)
            candidates.append(candidate)
        return min(candidates, key=self._evaluate_circuit)

    def run_iteration(self, iteration: int) -> List[Dict[str, Any]]:
        params = self._optimize_params(iteration)
        counts = self._sample_counts(params)
        self.sampling_calls += 1
        self.sampling_shots += sum(counts.values())
        top_samples = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:8]

        failures: List[Dict[str, Any]] = []
        for bitstring, count in top_samples:
            evaluation = evaluate_state(decode_bitstring(bitstring))
            self.n_env_evaluations_total += 1
            self._env_eval_unique_keys.add(_state_key(evaluation["params"]))
            evaluation["iteration"] = iteration
            evaluation["bitstring"] = bitstring
            evaluation["count"] = count
            if evaluation["failure"]:
                failures.append(evaluation)
                key = _state_key(evaluation["params"])
                if key not in self._known_keys:
                    self._known_keys.add(key)
                    self.all_failures.append(evaluation)

        self.iteration_counts.append(len(self.all_failures))
        return failures

    def run(self, k_iterations: int = 50, output_path: str | Path | None = None, force: bool = False) -> Dict[str, Any]:
        if output_path is not None:
            checked_path = Path(output_path)
            if not checked_path.is_absolute():
                checked_path = ROOT / checked_path
            if checked_path.exists() and not force:
                raise FileExistsError(f"Refusing to overwrite {checked_path}")
        started = time.time()
        for iteration in range(k_iterations):
            self.run_iteration(iteration)

        results = {
            "method": "QAOA",
            "backend": self.backend_name,
            "used_qiskit": QISKIT_AVAILABLE,
            "n_qubits": N_QUBITS,
            "reps": self.reps,
            "shots": self.shots,
            "k_iterations": k_iterations,
            "elapsed_seconds": round(time.time() - started, 3),
            "total_unique_failures": len(self.all_failures),
            "n_env_evaluations_total": self.n_env_evaluations_total,
            "n_env_evaluations_unique": len(self._env_eval_unique_keys),
            "objective_convention": "pair_once_symmetric_storage",
            "resources": {
                "optimizer_evaluations": self.optimizer_evaluations,
                "optimizer_nfev_per_iteration": self.optimizer_nfev,
                "optimizer_shots": self.optimizer_shots,
                "sampling_calls": self.sampling_calls,
                "sampling_shots": self.sampling_shots,
                "circuit_shots_total": self.optimizer_shots + self.sampling_shots if QISKIT_AVAILABLE else 0,
                "fallback_draws_total": 0 if QISKIT_AVAILABLE else self.optimizer_shots + self.sampling_shots,
                "setup_catalog_calls": len(self._evaluated_states),
                "search_simulator_calls": self.n_env_evaluations_total,
                "search_unique_states": len(self._env_eval_unique_keys),
                "reporting_simulator_calls": 0,
                "simulator_calls_total_this_run": len(self._evaluated_states) + self.n_env_evaluations_total,
                "simulator_unique_states_this_run": len(self._evaluated_states) if self._evaluated_states else len(self._env_eval_unique_keys),
                "training_label_acquisition_calls_this_run": 0,
                "prior_training_cost": "Not measured here: reads persisted phase1 failure labels; NOT label-free. See input hash and historical training protocol.",
                "configured_optimizer_shots_upper_budget": k_iterations * 35 * self.shots,
            },
            "env_evaluation_note": (
                "run_iteration() evaluates the top-8 bitstrings each iteration, so the "
                "exploration loop performs up to 8*k non-unique environment evaluations "
                "over k iterations; this is distinct from the per-iteration COBYLA optimizer "
                "budget; the 256-state catalog is now built only for the non-Qiskit fallback."
            ),
            "cumulative_failures": self.iteration_counts,
            "failures": self.all_failures,
            "circuit_info": {
                "num_qubits": self.circuit.num_qubits,
                "depth": self.circuit.depth(),
                "num_params": self.circuit.num_parameters,
                "reps": self.reps,
                "backend": self.backend_name,
            },
            "circuit_text": str(self.circuit.draw(output="text", fold=80)),
        }

        if output_path is not None:
            out_path = Path(output_path)
            if not out_path.is_absolute():
                out_path = ROOT / out_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QAA Phase 2 - QAOA exploration runner")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=str(OUT / "qaoa_results.json"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _safe_draw_text(circuit: Any) -> str:
    text = str(circuit.draw(output="text", fold=80))
    return text.encode("ascii", errors="replace").decode("ascii")


def main() -> None:
    args = build_parser().parse_args()
    explorer = QAOAExplorer(reps=args.reps, shots=args.shots, seed=args.seed)
    print("=" * 56)
    print("QAA Phase 2 - QAOA Exploration")
    print("=" * 56)
    print(f"backend   : {explorer.backend_name}")
    print(f"reps      : {args.reps}")
    print(f"shots     : {args.shots}")
    print(f"iterations: {args.k}")
    print(f"num_qubits: {explorer.circuit.num_qubits}")
    print(f"depth     : {explorer.circuit.depth()}")

    if args.dry_run:
        print("\nDry run complete - circuit built successfully")
        print(_safe_draw_text(explorer.circuit))
        return

    results = explorer.run(k_iterations=args.k, output_path=args.output, force=args.force)
    print(f"\nUnique failures found: {results['total_unique_failures']}")
    print(f"Results written to   : {args.output}")


if __name__ == "__main__":
    main()
