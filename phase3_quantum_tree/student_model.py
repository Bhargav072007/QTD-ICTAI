"""
Autonomous student model trained on distilled quantum tree targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class AutonomousStudentModel:
    """
    Single-layer logistic regression (no hidden layers) trained on distilled targets.

    Architecture: sigmoid(X @ W + b) where X is (n_samples, 4), W is (4, 1).
    This is logistic regression, not a multi-layer neural network.
    The paper describes this as a 'distilled student model'; the architecture
    is intentionally lightweight for interpretability and speed.
    """

    input_dim: int = 4
    learning_rate: float = 0.05
    epochs: int = 350
    seed: int = 42

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(0.0, 0.15, size=(self.input_dim, 1))
        self.bias = np.zeros((1, 1), dtype=float)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

    def fit(self, features: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        y = targets.reshape(-1, 1)
        for _ in range(self.epochs):
            logits = features @ self.weights + self.bias
            preds = self._sigmoid(logits)
            error = preds - y
            self.weights -= self.learning_rate * (features.T @ error) / len(features)
            self.bias -= self.learning_rate * np.mean(error, axis=0, keepdims=True)

        preds = self.predict(features)
        mse = float(np.mean(np.square(preds - y)))
        return {"mse": round(mse, 6)}

    def predict(self, features: np.ndarray) -> np.ndarray:
        logits = features @ self.weights + self.bias
        return self._sigmoid(logits)
