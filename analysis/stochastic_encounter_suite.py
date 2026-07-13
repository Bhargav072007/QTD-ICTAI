"""Materialize a locked synthetic, aviation-inspired stochastic encounter suite.

The generator is deliberately explicit about being synthetic.  It models a
nonlinear failure probability and observes independent Bernoulli rollout counts
for every scenario.  It does not model a certified aircraft, controller, or
real-world encounter distribution.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES = ("heading_deg", "altitude_ft", "speed_nm_step", "lateral_nm", "response_delay_s", "wind_nm_step")


@dataclass(frozen=True)
class Regime:
    name: str
    seed: int
    heading_center: float
    heading_sd: float
    altitude_center: float
    altitude_sd: float
    speed_center: float
    delay_center: float
    wind_center: float
    bias: float
    hazard_heading: float
    hazard_delay: float
    hazard_wind: float


REGIMES = (
    Regime("nominal", 701, 95.0, 16.0, 0.0, 1250.0, 4.5, 2.0, 0.0, -5.4, 83.0, 3.2, 0.0),
    Regime("crosswind_shift", 702, 88.0, 20.0, 0.0, 1350.0, 4.7, 2.2, 0.7, -5.2, 80.0, 3.0, 0.9),
    Regime("late_response", 703, 96.0, 16.0, 0.0, 1250.0, 4.6, 4.5, 0.0, -5.3, 84.0, 2.4, 0.0),
    Regime("altitude_shift", 704, 92.0, 18.0, 450.0, 1550.0, 4.8, 2.5, -0.3, -5.1, 81.0, 3.0, -0.4),
)


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -35.0, 35.0)))


def sample_features(regime: Regime, count: int, rng: np.random.Generator) -> np.ndarray:
    heading = np.clip(rng.normal(regime.heading_center, regime.heading_sd, count), 55.0, 135.0)
    altitude = np.clip(rng.normal(regime.altitude_center, regime.altitude_sd, count), -4000.0, 4000.0)
    speed = np.clip(rng.normal(regime.speed_center, 0.9, count), 2.0, 7.5)
    lateral = rng.uniform(-30.0, 30.0, count)
    delay = np.clip(rng.normal(regime.delay_center, 1.2, count), 0.0, 8.0)
    wind = np.clip(rng.normal(regime.wind_center, 0.65, count), -2.5, 2.5)
    return np.column_stack((heading, altitude, speed, lateral, delay, wind))


def failure_probability(features: np.ndarray, regime: Regime) -> np.ndarray:
    """Hidden nonlinear stochastic-oracle probability for a candidate encounter."""
    heading, altitude, speed, lateral, delay, wind = features.T
    alignment = np.exp(-((heading - regime.hazard_heading) / 12.0) ** 2)
    lateral_overlap = np.exp(-(lateral / 8.5) ** 2)
    altitude_overlap = np.exp(-(altitude / 850.0) ** 2)
    speed_pressure = sigmoid((speed - 4.5) * 1.5)
    delay_pressure = sigmoid((delay - regime.hazard_delay) * 1.1)
    wind_alignment = np.exp(-((wind - regime.hazard_wind) / 0.8) ** 2)
    interaction = alignment * lateral_overlap * altitude_overlap
    logit = (
        regime.bias
        + 4.6 * interaction
        + 1.0 * speed_pressure
        + 1.3 * delay_pressure
        + 0.8 * wind_alignment
        + 1.2 * interaction * delay_pressure
        + 0.7 * interaction * wind_alignment
    )
    return sigmoid(logit)


def normalize_features(features: np.ndarray) -> np.ndarray:
    lower = np.asarray((55.0, -4000.0, 2.0, -30.0, 0.0, -2.5))
    upper = np.asarray((135.0, 4000.0, 7.5, 30.0, 8.0, 2.5))
    return 2.0 * ((features - lower) / (upper - lower)) - 1.0


def observe_rollouts(probabilities: np.ndarray, rollouts: int, rng: np.random.Generator) -> np.ndarray:
    return rng.binomial(rollouts, probabilities).astype(int)


def scenario_rows(features: np.ndarray, probabilities: np.ndarray, observed_failures: np.ndarray, rollouts: int, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (feature_row, probability, failures) in enumerate(zip(features, probabilities, observed_failures)):
        rows.append({
            "scenario_id": f"{split}-{index:04d}",
            "features": {name: round(float(value), 8) for name, value in zip(FEATURE_NAMES, feature_row)},
            "normalized_features": [round(float(value), 8) for value in normalize_features(feature_row.reshape(1, -1))[0]],
            "hidden_failure_probability": round(float(probability), 10),
            "observed_rollouts": rollouts,
            "observed_failure_count": int(failures),
            "observed_failure_rate": round(float(failures / rollouts), 10),
        })
    return rows


def materialize_regime(regime: Regime, train_size: int, test_size: int, train_rollouts: int, test_rollouts: int) -> dict[str, Any]:
    rng = np.random.default_rng(regime.seed)
    train_features = sample_features(regime, train_size, rng)
    test_features = sample_features(regime, test_size, rng)
    train_probability = failure_probability(train_features, regime)
    test_probability = failure_probability(test_features, regime)
    train_failures = observe_rollouts(train_probability, train_rollouts, rng)
    test_failures = observe_rollouts(test_probability, test_rollouts, rng)
    return {
        "regime": asdict(regime),
        "train": scenario_rows(train_features, train_probability, train_failures, train_rollouts, "train"),
        "test": scenario_rows(test_features, test_probability, test_failures, test_rollouts, "test"),
        "summary": {
            "train_mean_hidden_failure_probability": round(float(train_probability.mean()), 8),
            "test_mean_hidden_failure_probability": round(float(test_probability.mean()), 8),
            "train_positive_observed_scenarios": int(np.sum(train_failures > 0)),
            "test_positive_observed_scenarios": int(np.sum(test_failures > 0)),
            "train_total_observed_failures": int(train_failures.sum()),
            "test_total_observed_failures": int(test_failures.sum()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the locked synthetic stochastic encounter suite")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "stochastic_suite" / "suite_v1.json")
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--train-rollouts", type=int, default=32)
    parser.add_argument("--test-rollouts", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable suite output: {args.output}")
    if min(args.train_size, args.test_size, args.train_rollouts, args.test_rollouts) < 1:
        raise ValueError("all sizes and rollout counts must be positive")
    regimes = [materialize_regime(regime, args.train_size, args.test_size, args.train_rollouts, args.test_rollouts) for regime in REGIMES]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite_name": "synthetic_aviation_inspired_stochastic_encounter_v1",
        "scope": "synthetic benchmark only; not operational aviation validation or certification evidence",
        "feature_names": FEATURE_NAMES,
        "protocol": "analysis/STOCHASTIC_SUITE_PROTOCOL.md",
        "regimes": regimes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for regime in regimes:
        summary = regime["summary"]
        print(
            f"{regime['regime']['name']}: test mean risk={summary['test_mean_hidden_failure_probability']:.4f} "
            f"test positive scenarios={summary['test_positive_observed_scenarios']}/{args.test_size} "
            f"test observed failures={summary['test_total_observed_failures']}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
