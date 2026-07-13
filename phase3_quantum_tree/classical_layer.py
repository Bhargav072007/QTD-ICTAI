"""
Layer 1: classical teacher simulation with backpropagation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from phase2_qaoa.qaoa_runner import evaluate_state
from phase2_qaoa.qubo_encoder import enumerate_parameter_states


FEATURE_NAMES = ("int_heading", "int_altitude", "int_speed", "int_x_offset")


def _normalize_state(state: Dict[str, float]) -> np.ndarray:
    return np.array(
        [
            (state["int_heading"] - 97.5) / 22.5,
            (state["int_altitude"] - 30000.0) / 2500.0,
            (state["int_speed"] - 4.5) / 1.5,
            state["int_x_offset"] / 20.0,
        ],
        dtype=float,
    )


Evaluator = Callable[[Dict[str, float]], Dict[str, Any]]


def build_classical_dataset(
    evaluator: Evaluator = evaluate_state,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    """Builds a supervised dataset from the Phase 2 encounter evaluator."""
    states = enumerate_parameter_states()
    features = np.stack([_normalize_state(state) for state in states], axis=0)
    labels = np.array(
        [
            1.0
            if evaluator(state)["failure"]
            else 0.0
            for state in states
        ],
        dtype=float,
    ).reshape(-1, 1)
    return features, labels, states


@dataclass
class ClassicalTeacherModel:
    """Small MLP trained with manual backprop so the layer is self-contained."""

    input_dim: int = 4
    hidden_dim: int = 8
    learning_rate: float = 0.08
    epochs: int = 400
    seed: int = 42

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.w1 = rng.normal(0.0, 0.35, size=(self.input_dim, self.hidden_dim))
        self.b1 = np.zeros((1, self.hidden_dim), dtype=float)
        self.w2 = rng.normal(0.0, 0.25, size=(self.hidden_dim, 1))
        self.b2 = np.zeros((1, 1), dtype=float)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

    def fit(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        for _ in range(self.epochs):
            z1 = x @ self.w1 + self.b1
            a1 = np.tanh(z1)
            z2 = a1 @ self.w2 + self.b2
            y_hat = self._sigmoid(z2)

            error = y_hat - y
            grad_w2 = (a1.T @ error) / len(x)
            grad_b2 = np.mean(error, axis=0, keepdims=True)
            hidden_error = (error @ self.w2.T) * (1.0 - np.square(a1))
            grad_w1 = (x.T @ hidden_error) / len(x)
            grad_b1 = np.mean(hidden_error, axis=0, keepdims=True)

            self.w2 -= self.learning_rate * grad_w2
            self.b2 -= self.learning_rate * grad_b2
            self.w1 -= self.learning_rate * grad_w1
            self.b1 -= self.learning_rate * grad_b1

        preds = self.predict_proba(x)
        loss = float(np.mean(-(y * np.log(preds + 1e-8) + (1 - y) * np.log(1 - preds + 1e-8))))
        accuracy = float(np.mean((preds >= 0.5) == y))
        return {"loss": round(loss, 6), "accuracy": round(accuracy, 6)}

    def hidden_representation(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x @ self.w1 + self.b1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        hidden = self.hidden_representation(x)
        return self._sigmoid(hidden @ self.w2 + self.b2)


def run_classical_layer(
    seed: int = 42,
    evaluator: Evaluator = evaluate_state,
) -> Dict[str, object]:
    x, y, states = build_classical_dataset(evaluator=evaluator)
    teacher = ClassicalTeacherModel(seed=seed)
    metrics = teacher.fit(x, y)
    hidden = teacher.hidden_representation(x)
    probs = teacher.predict_proba(x).reshape(-1)
    return {
        "features": x,
        "labels": y.reshape(-1),
        "states": states,
        "hidden": hidden,
        "teacher_probs": probs,
        "teacher_metrics": metrics,
        "teacher": teacher,
    }
