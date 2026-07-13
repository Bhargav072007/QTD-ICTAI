"""Pre-registered QTDv2 versus matched-classical held-out ranking benchmark.

The runner is intentionally separate from the manuscript pipeline.  It never
overwrites canonical outputs and records every outer/inner split and ranking.
It tests an empirical feature-map comparison; it cannot establish quantum
computational advantage because its quantum feature map is statevector-simulated.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss
from sklearn.preprocessing import StandardScaler

from phase2_qaoa.qaoa_runner import evaluate_state
from phase2_qaoa.qubo_encoder import enumerate_parameter_states
from phase3_quantum_tree.classical_layer import _normalize_state
from run_split_experiment import auc, state_key, stratified_split


ROOT = Path(__file__).resolve().parent
DEFAULT_SEEDS = list(range(42, 52))
K_VALUES = (5, 10, 20, 30, 50)
N_QUBITS = 4
LAYERS = 2
PARAMETER_COUNT = LAYERS * 2 * N_QUBITS


def pstd(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": round(float(statistics.mean(values)), 6),
        "std": round(float(statistics.pstdev(values)), 6),
    }


def bootstrap_mean_ci(values: Sequence[float], seed: int = 0) -> dict[str, float | int]:
    raw = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(raw, size=(10_000, len(raw)), replace=True).mean(axis=1)
    return {
        "mean": round(float(raw.mean()), 6),
        "ci_95_low": round(float(np.quantile(samples, 0.025)), 6),
        "ci_95_high": round(float(np.quantile(samples, 0.975)), 6),
        "resamples": 10_000,
        "seed": seed,
    }


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def inner_split(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Make a stratified 75/25 partition of an outer-training split."""
    rng = np.random.default_rng(seed)
    fit: list[int] = []
    tune: list[int] = []
    for label in (0.0, 1.0):
        indices = rng.permutation(np.flatnonzero(labels == label))
        cut = max(1, int(math.floor(len(indices) * 0.75)))
        cut = min(cut, len(indices) - 1)
        fit.extend(int(index) for index in indices[:cut])
        tune.extend(int(index) for index in indices[cut:])
    return np.asarray(sorted(fit), dtype=int), np.asarray(sorted(tune), dtype=int)


def _apply_ry(state: np.ndarray, qubit: int, angle: float) -> None:
    half = angle / 2.0
    c, s = math.cos(half), math.sin(half)
    mask = 1 << qubit
    for index in range(len(state)):
        if index & mask:
            continue
        paired = index | mask
        zero, one = state[index], state[paired]
        state[index] = c * zero - s * one
        state[paired] = s * zero + c * one


def _apply_rz(state: np.ndarray, qubit: int, angle: float) -> None:
    phase_zero = np.exp(-0.5j * angle)
    phase_one = np.exp(0.5j * angle)
    mask = 1 << qubit
    for index in range(len(state)):
        state[index] *= phase_one if index & mask else phase_zero


def _apply_cz(state: np.ndarray, left: int, right: int) -> None:
    mask = (1 << left) | (1 << right)
    for index in range(len(state)):
        if (index & mask) == mask:
            state[index] *= -1.0


def quantum_features(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Return Z and nearest-neighbour ZZ expectations of a four-qubit circuit."""
    params = params.reshape(LAYERS, 2, N_QUBITS)
    dimension = 1 << N_QUBITS
    state = np.zeros((len(x), dimension), dtype=complex)
    state[:, 0] = 1.0
    basis = np.arange(dimension)
    for layer in range(LAYERS):
        for qubit in range(N_QUBITS):
            ry_angles = math.pi * x[:, qubit] + params[layer, 0, qubit]
            c = np.cos(ry_angles / 2.0)
            s = np.sin(ry_angles / 2.0)
            mask = 1 << qubit
            for index in range(dimension):
                if index & mask:
                    continue
                paired = index | mask
                zero, one = state[:, index].copy(), state[:, paired].copy()
                state[:, index] = c * zero - s * one
                state[:, paired] = s * zero + c * one
            rz_angles = math.pi * x[:, qubit] + params[layer, 1, qubit]
            phase_zero = np.exp(-0.5j * rz_angles)
            phase_one = np.exp(0.5j * rz_angles)
            for index in range(dimension):
                state[:, index] *= phase_one if (index & mask) else phase_zero
        for qubit in range(N_QUBITS - 1):
            both_one = ((basis & (1 << qubit)) != 0) & ((basis & (1 << (qubit + 1))) != 0)
            state[:, both_one] *= -1.0
        ring_both_one = ((basis & 1) != 0) & ((basis & (1 << (N_QUBITS - 1))) != 0)
        state[:, ring_both_one] *= -1.0
    probabilities = np.abs(state) ** 2
    z_signs = np.asarray([[1.0 if ((index >> qubit) & 1) == 0 else -1.0 for index in basis] for qubit in range(N_QUBITS)])
    zz_signs = np.asarray([
        [1.0 if ((index >> qubit) & 1) == ((index >> ((qubit + 1) % N_QUBITS)) & 1) else -1.0 for index in basis]
        for qubit in range(N_QUBITS)
    ])
    return np.concatenate((probabilities @ z_signs.T, probabilities @ zz_signs.T), axis=1)


def classical_features(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Matched-size trainable classical trigonometric feature map."""
    params = params.reshape(LAYERS, 2, N_QUBITS)
    features: list[np.ndarray] = []
    for layer in range(LAYERS):
        features.append(np.sin(math.pi * x + params[layer, 0]))
        features.append(np.cos(math.pi * x + params[layer, 1]))
    full = np.concatenate(features, axis=1)
    # Eight output features, matching the number of quantum observables.
    return 0.5 * (full[:, : 2 * N_QUBITS] + full[:, 2 * N_QUBITS :])


FeatureMap = Callable[[np.ndarray, np.ndarray], np.ndarray]


def fit_predict(features: FeatureMap, params: np.ndarray, x_fit: np.ndarray, y_fit: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    z_fit = scaler.fit_transform(features(x_fit, params))
    z_eval = scaler.transform(features(x_eval, params))
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2_000, random_state=0)
    model.fit(z_fit, y_fit)
    return model.predict_proba(z_eval)[:, 1]


def tune_parameters(
    feature_map: FeatureMap,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_tune: np.ndarray,
    y_tune: np.ndarray,
    seed: int,
    steps: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """SPSA selects parameters using only an inner validation partition."""
    rng = np.random.default_rng(seed)
    current = rng.normal(0.0, 0.05, PARAMETER_COUNT)
    best = current.copy()

    def objective(params: np.ndarray) -> float:
        probabilities = np.clip(fit_predict(feature_map, params, x_fit, y_fit, x_tune), 1e-6, 1 - 1e-6)
        return float(log_loss(y_tune, probabilities, labels=[0.0, 1.0]))

    best_loss = objective(best)
    history = [best_loss]
    for step in range(steps):
        perturbation = rng.choice((-1.0, 1.0), size=PARAMETER_COUNT)
        c = 0.15 / ((step + 1) ** 0.101)
        a = 0.08 / ((step + 11) ** 0.602)
        plus = objective(current + c * perturbation)
        minus = objective(current - c * perturbation)
        gradient = ((plus - minus) / (2.0 * c)) * perturbation
        current = np.clip(current - a * gradient, -math.pi, math.pi)
        loss = objective(current)
        history.append(loss)
        if loss < best_loss:
            best, best_loss = current.copy(), loss
    return best, {"initial_or_step_losses": [round(float(value), 8) for value in history], "best_inner_log_loss": round(float(best_loss), 8)}


def ranking_metrics(scores: np.ndarray, labels: np.ndarray, state_keys: Sequence[tuple[float, ...]]) -> dict[str, Any]:
    order = np.argsort(scores)[::-1]
    ranked_labels = labels[order]
    total_failures = int(labels.sum())
    cumulative = np.cumsum(ranked_labels).astype(int)
    by_k: dict[str, Any] = {}
    for k in K_VALUES:
        selected = ranked_labels[:k]
        found = int(selected.sum())
        by_k[str(k)] = {
            "found": found,
            "recall": round(found / total_failures, 6) if total_failures else 0.0,
            "precision": round(found / k, 6),
            "discovery_auc": auc(cumulative[:k]),
        }
    return {
        "average_precision": round(float(average_precision_score(labels, scores)), 6),
        "ranking_order": [list(state_keys[int(index)]) for index in order],
        "metrics_at_k": by_k,
    }


def run_method(
    name: str,
    feature_map: FeatureMap,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_keys: Sequence[tuple[float, ...]],
    seed: int,
    spsa_steps: int,
) -> dict[str, Any]:
    fit_indices, tune_indices = inner_split(y_train, seed + 10_000)
    params, tuning = tune_parameters(
        feature_map, x_train[fit_indices], y_train[fit_indices], x_train[tune_indices], y_train[tune_indices], seed, spsa_steps
    )
    scores = fit_predict(feature_map, params, x_train, y_train, x_test)
    return {
        "method": name,
        "inner_fit_indices": fit_indices.tolist(),
        "inner_tune_indices": tune_indices.tolist(),
        "parameters": [round(float(value), 8) for value in params],
        "tuning": tuning,
        "test_scores": [round(float(value), 8) for value in scores],
        **ranking_metrics(scores, y_test, test_keys),
    }


def raw_logistic(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, test_keys: Sequence[tuple[float, ...]]
) -> dict[str, Any]:
    scaler = StandardScaler()
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2_000, random_state=0)
    scores = model.fit(scaler.fit_transform(x_train), y_train).predict_proba(scaler.transform(x_test))[:, 1]
    return {"method": "raw_logistic", "test_scores": [round(float(value), 8) for value in scores], **ranking_metrics(scores, y_test, test_keys)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the locked QTDv2 matched-classical benchmark")
    parser.add_argument("--seeds", type=int, default=10, help="Number of locked outer seeds, starting at 42.")
    parser.add_argument("--spsa-steps", type=int, default=25, help="Predeclared SPSA steps for each method and split.")
    parser.add_argument("--output-dir", type=Path, default=None, help="New directory for output; defaults to a timestamped reproduction directory.")
    args = parser.parse_args()
    if args.seeds < 2 or args.seeds > len(DEFAULT_SEEDS):
        raise ValueError("--seeds must be between 2 and 10 to preserve the locked seed family")
    if args.spsa_steps < 1:
        raise ValueError("--spsa-steps must be positive")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or ROOT / "outputs" / "reproduction" / f"qtd_v2_matched_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "qtd_v2_matched_benchmark.json"

    states = enumerate_parameter_states()
    labels = np.asarray([1.0 if evaluate_state(state)["failure"] else 0.0 for state in states], dtype=float)
    x = np.stack([_normalize_state(state) for state in states], axis=0)
    seeds = DEFAULT_SEEDS[: args.seeds]
    records: list[dict[str, Any]] = []
    for seed in seeds:
        train_indices, test_indices = stratified_split(states, labels, seed)
        train_keys = [state_key(states[index]) for index in train_indices]
        test_keys = [state_key(states[index]) for index in test_indices]
        if set(train_keys) & set(test_keys):
            raise RuntimeError("outer train/test leakage")
        methods = {
            "raw_logistic": raw_logistic(x[train_indices], labels[train_indices], x[test_indices], labels[test_indices], test_keys),
            "qtd_v2_trainable": run_method("qtd_v2_trainable", quantum_features, x[train_indices], labels[train_indices], x[test_indices], labels[test_indices], test_keys, seed, args.spsa_steps),
            "matched_classical_trig": run_method("matched_classical_trig", classical_features, x[train_indices], labels[train_indices], x[test_indices], labels[test_indices], test_keys, seed, args.spsa_steps),
        }
        records.append({
            "outer_seed": seed,
            "train_state_keys": [list(key) for key in train_keys],
            "test_state_keys": [list(key) for key in test_keys],
            "train_failure_count": int(labels[train_indices].sum()),
            "test_failure_count": int(labels[test_indices].sum()),
            "methods": methods,
        })
        q_auc = methods["qtd_v2_trainable"]["metrics_at_k"]["50"]["discovery_auc"]
        c_auc = methods["matched_classical_trig"]["metrics_at_k"]["50"]["discovery_auc"]
        print(f"seed={seed} qtd_auc={q_auc} classical_auc={c_auc} delta={q_auc-c_auc:.3f}")

    q_auc = [row["methods"]["qtd_v2_trainable"]["metrics_at_k"]["50"]["discovery_auc"] for row in records]
    c_auc = [row["methods"]["matched_classical_trig"]["metrics_at_k"]["50"]["discovery_auc"] for row in records]
    q_ap = [row["methods"]["qtd_v2_trainable"]["average_precision"] for row in records]
    c_ap = [row["methods"]["matched_classical_trig"]["average_precision"] for row in records]
    deltas = [float(q) - float(c) for q, c in zip(q_auc, c_auc)]
    ap_deltas = [float(q) - float(c) for q, c in zip(q_ap, c_ap)]
    auc_ci = bootstrap_mean_ci(deltas)
    decision = {
        "primary_auc_delta": {"per_seed": deltas, "aggregate": pstd(deltas), "bootstrap_mean_95_ci": auc_ci},
        "average_precision_delta": {"per_seed": ap_deltas, "aggregate": pstd(ap_deltas)},
        "positive_auc_delta_count": sum(delta > 0 for delta in deltas),
        "empirical_improvement_rule_passed": bool(
            auc_ci["ci_95_low"] > 0 and sum(delta > 0 for delta in deltas) >= 8 and statistics.mean(ap_deltas) > 0
        ),
        "rule": "CI lower bound > 0; at least 8/10 positive AUC deltas; positive mean AP delta.",
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(),
        "protocol": "analysis/QTDV2_MATCHED_PROTOCOL.md",
        "scope": "within-benchmark statevector-simulation feature-map comparison; not quantum advantage",
        "config": {"outer_seeds": seeds, "spsa_steps": args.spsa_steps, "qubits": N_QUBITS, "layers": LAYERS, "trainable_parameter_count": PARAMETER_COUNT, "k_values": K_VALUES},
        "records": records,
        "decision": decision,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
