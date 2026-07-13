"""Fair trainable-QTD ranking evaluation on the locked stochastic suite.

This runner deliberately ignores hidden oracle probabilities.  Models see only
training rollout rates; test rollout rates are read only after fitting and
inner-tuning are complete.  It is a statevector-simulation comparison, not a
quantum-advantage experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
N_QUBITS = 6
LAYERS = 2
PARAMETERS = N_QUBITS * LAYERS * 2
K_VALUES = (10, 25, 50, 100, 200)


def pstd(values: Sequence[float]) -> dict[str, float]:
    return {"mean": round(float(statistics.mean(values)), 8), "std": round(float(statistics.pstdev(values)), 8)}


def bootstrap_mean_ci(values: Sequence[float]) -> dict[str, float | int]:
    source = np.asarray(values, dtype=float)
    rng = np.random.default_rng(0)
    means = rng.choice(source, size=(10_000, len(source)), replace=True).mean(axis=1)
    return {
        "mean": round(float(source.mean()), 8),
        "ci_95_low": round(float(np.quantile(means, 0.025)), 8),
        "ci_95_high": round(float(np.quantile(means, 0.975)), 8),
        "resamples": 10_000,
        "seed": 0,
    }


def inner_split(size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(size)
    cut = int(math.floor(size * 0.75))
    return np.sort(indices[:cut]), np.sort(indices[cut:])


def quantum_features(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Six-qubit RY/RZ/CZ feature map with Z and ring-ZZ expectations."""
    params = params.reshape(LAYERS, 2, N_QUBITS)
    dimension = 1 << N_QUBITS
    state = np.zeros((len(x), dimension), dtype=complex)
    state[:, 0] = 1.0
    basis = np.arange(dimension)
    for layer in range(LAYERS):
        for qubit in range(N_QUBITS):
            ry = math.pi * x[:, qubit] + params[layer, 0, qubit]
            c, s = np.cos(ry / 2.0), np.sin(ry / 2.0)
            mask = 1 << qubit
            for index in range(dimension):
                if index & mask:
                    continue
                paired = index | mask
                zero, one = state[:, index].copy(), state[:, paired].copy()
                state[:, index] = c * zero - s * one
                state[:, paired] = s * zero + c * one
            rz = math.pi * x[:, qubit] + params[layer, 1, qubit]
            phase_zero, phase_one = np.exp(-0.5j * rz), np.exp(0.5j * rz)
            for index in range(dimension):
                state[:, index] *= phase_one if index & mask else phase_zero
        for qubit in range(N_QUBITS):
            right = (qubit + 1) % N_QUBITS
            both_one = ((basis & (1 << qubit)) != 0) & ((basis & (1 << right)) != 0)
            state[:, both_one] *= -1.0
    probabilities = np.abs(state) ** 2
    z = np.asarray([[1.0 if ((index >> qubit) & 1) == 0 else -1.0 for index in basis] for qubit in range(N_QUBITS)])
    zz = np.asarray([
        [1.0 if ((index >> qubit) & 1) == ((index >> ((qubit + 1) % N_QUBITS)) & 1) else -1.0 for index in basis]
        for qubit in range(N_QUBITS)
    ])
    return np.concatenate((probabilities @ z.T, probabilities @ zz.T), axis=1)


def classical_features(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Twelve-output classical map with identical trainable-offset count."""
    params = params.reshape(LAYERS, 2, N_QUBITS)
    first = np.sin(math.pi * x + params[0, 0]) + np.cos(math.pi * x + params[0, 1])
    second = np.sin(math.pi * x + params[1, 0]) + np.cos(math.pi * x + params[1, 1])
    return np.concatenate((first, second), axis=1) / 2.0


FeatureMap = Callable[[np.ndarray, np.ndarray], np.ndarray]


def fit_predict(feature_map: FeatureMap, params: np.ndarray, x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    train_features = scaler.fit_transform(feature_map(x_train, params))
    eval_features = scaler.transform(feature_map(x_eval, params))
    model = Ridge(alpha=1.0)
    return np.clip(model.fit(train_features, y_train).predict(eval_features), 0.0, 1.0)


def tune(feature_map: FeatureMap, x: np.ndarray, y: np.ndarray, seed: int, steps: int) -> tuple[np.ndarray, dict[str, Any]]:
    fit_indices, tune_indices = inner_split(len(x), seed + 31_337)
    x_fit, y_fit, x_tune, y_tune = x[fit_indices], y[fit_indices], x[tune_indices], y[tune_indices]
    rng = np.random.default_rng(seed)
    current = rng.normal(0.0, 0.05, PARAMETERS)
    best = current.copy()

    def objective(params: np.ndarray) -> float:
        return float(mean_squared_error(y_tune, fit_predict(feature_map, params, x_fit, y_fit, x_tune)))

    best_loss = objective(best)
    losses = [best_loss]
    for step in range(steps):
        direction = rng.choice((-1.0, 1.0), size=PARAMETERS)
        c = 0.14 / ((step + 1) ** 0.101)
        a = 0.07 / ((step + 11) ** 0.602)
        plus, minus = objective(current + c * direction), objective(current - c * direction)
        current = np.clip(current - a * ((plus - minus) / (2.0 * c)) * direction, -math.pi, math.pi)
        loss = objective(current)
        losses.append(loss)
        if loss < best_loss:
            best, best_loss = current.copy(), loss
    return best, {
        "inner_fit_indices": fit_indices.tolist(),
        "inner_tune_indices": tune_indices.tolist(),
        "losses": [round(float(loss), 10) for loss in losses],
        "best_inner_mse": round(float(best_loss), 10),
    }


def metrics(scores: np.ndarray, relevance: np.ndarray, ids: Sequence[str]) -> dict[str, Any]:
    order = np.argsort(scores)[::-1]
    ranked = relevance[order]
    total = float(relevance.sum())
    cumulative = np.cumsum(ranked)
    outcomes: dict[str, Any] = {}
    for k in K_VALUES:
        mass = float(cumulative[k - 1])
        normalized_auc = float(np.trapezoid(np.concatenate(([0.0], cumulative[:k] / total)), dx=1.0) / k) if total else 0.0
        outcomes[str(k)] = {
            "risk_mass_recall": round(mass / total, 8) if total else 0.0,
            "mean_observed_failure_rate": round(float(ranked[:k].mean()), 8),
            "normalized_discovery_auc": round(normalized_auc, 8),
        }
    return {
        "ranking_order": [ids[int(index)] for index in order],
        "metrics_at_k": outcomes,
        "test_rate_mse": round(float(mean_squared_error(relevance, scores)), 8),
    }


def evaluate_method(name: str, feature_map: FeatureMap, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, ids: Sequence[str], seed: int, steps: int) -> dict[str, Any]:
    params, tuning = tune(feature_map, x_train, y_train, seed, steps)
    scores = fit_predict(feature_map, params, x_train, y_train, x_test)
    return {
        "method": name,
        "parameters": [round(float(value), 8) for value in params],
        "tuning": tuning,
        "test_scores": [round(float(score), 8) for score in scores],
        **metrics(scores, y_test, ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QTDv3 versus matched classical features on the locked stochastic suite")
    parser.add_argument("--suite", type=Path, default=ROOT / "outputs" / "stochastic_suite" / "suite_v1.json")
    parser.add_argument("--spsa-steps", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or ROOT / "outputs" / "reproduction" / f"qtd_v3_stochastic_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for regime_index, regime in enumerate(suite["regimes"]):
        train, test = regime["train"], regime["test"]
        x_train = np.asarray([row["normalized_features"] for row in train], dtype=float)
        y_train = np.asarray([row["observed_failure_rate"] for row in train], dtype=float)
        x_test = np.asarray([row["normalized_features"] for row in test], dtype=float)
        y_test = np.asarray([row["observed_failure_rate"] for row in test], dtype=float)
        ids = [str(row["scenario_id"]) for row in test]
        seed = int(regime["regime"]["seed"]) + regime_index
        quantum = evaluate_method("qtd_v3_trainable_statevector", quantum_features, x_train, y_train, x_test, y_test, ids, seed, args.spsa_steps)
        classical = evaluate_method("matched_classical_trig", classical_features, x_train, y_train, x_test, y_test, ids, seed, args.spsa_steps)
        records.append({"regime": regime["regime"]["name"], "methods": {"qtd_v3": quantum, "matched_classical": classical}})
        q_auc = quantum["metrics_at_k"]["200"]["normalized_discovery_auc"]
        c_auc = classical["metrics_at_k"]["200"]["normalized_discovery_auc"]
        print(f"{regime['regime']['name']}: qtd_auc={q_auc:.5f} classical_auc={c_auc:.5f} delta={q_auc-c_auc:.5f}")
    deltas = [row["methods"]["qtd_v3"]["metrics_at_k"]["200"]["normalized_discovery_auc"] - row["methods"]["matched_classical"]["metrics_at_k"]["200"]["normalized_discovery_auc"] for row in records]
    decision = {
        "primary_metric": "normalized discovery AUC at k=200 using hidden test rollout rates only after fitting",
        "per_regime_delta": deltas,
        "aggregate": pstd(deltas),
        "bootstrap_mean_95_ci": bootstrap_mean_ci(deltas),
        "positive_regime_count": sum(delta > 0 for delta in deltas),
        "interpretation": "Four synthetic regimes are insufficient for a quantum-advantage claim; report all results without selecting regimes.",
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite": str(args.suite),
        "protocol": "analysis/STOCHASTIC_SUITE_PROTOCOL.md",
        "scope": "synthetic aviation-inspired statevector evaluation; not operational aviation validation or quantum advantage",
        "config": {"qubits": N_QUBITS, "layers": LAYERS, "trainable_parameter_count": PARAMETERS, "spsa_steps": args.spsa_steps, "k_values": K_VALUES},
        "records": records,
        "decision": decision,
    }
    path = output_dir / "qtd_v3_stochastic_suite.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
