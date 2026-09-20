"""
Phase 1 - REINFORCE policy-gradient trainer for the aviation environment.
Algorithm: single-step REINFORCE (Williams 1992) — no baseline, no clipping.
This is NOT PPO. The paper should reference this as REINFORCE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase1.aviation_env_3d import ACTIONS, AviationEnv3D


OUT = ROOT / "outputs" / "phase1"
OUT.mkdir(parents=True, exist_ok=True)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    probs = np.exp(shifted)
    return probs / np.sum(probs)


class LinearPolicy:
    def __init__(self, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0.0, 0.08, size=(4, len(ACTIONS)))
        self.bias = np.zeros(len(ACTIONS), dtype=float)

    def action_probs(self, obs: np.ndarray) -> np.ndarray:
        logits = obs @ self.weights + self.bias
        return _softmax(logits)

    def act(self, obs: np.ndarray, rng: np.random.Generator) -> tuple[int, np.ndarray]:
        probs = self.action_probs(obs)
        action = int(rng.choice(len(ACTIONS), p=probs))
        return action, probs


def train_policy(
    seed: int = 42,
    epochs: int = 180,
    episodes_per_epoch: int = 64,
    learning_rate: float = 0.045,
) -> Dict[str, Any]:
    """Implements REINFORCE: gradient = (one_hot - probs) * reward, no advantage normalization,
    no value baseline, no clipping. Single-step episodes only."""
    env = AviationEnv3D(seed=seed)
    rng = np.random.default_rng(seed)
    policy = LinearPolicy(seed=seed)
    reward_trace = []
    rollout_calls = 0
    rollout_states = set()
    best_weights = policy.weights.copy()
    best_bias = policy.bias.copy()
    best_reward = float("-inf")

    for _ in range(epochs):
        gradients_w = np.zeros_like(policy.weights)
        gradients_b = np.zeros_like(policy.bias)
        rewards = []

        for _ in range(episodes_per_epoch):
            obs = env.reset()
            action, probs = policy.act(obs, rng)
            _, reward, _, outcome = env.step(action)
            rollout_calls += 1
            rollout_states.add(tuple(sorted(outcome["params"].items())))
            rewards.append(reward)

            one_hot = np.zeros(len(ACTIONS), dtype=float)
            one_hot[action] = 1.0
            advantage = reward
            gradients_w += np.outer(obs, (one_hot - probs) * advantage)
            gradients_b += (one_hot - probs) * advantage

        baseline = float(np.mean(rewards))
        reward_trace.append(round(baseline, 6))
        policy.weights += learning_rate * gradients_w / episodes_per_epoch
        policy.bias += learning_rate * gradients_b / episodes_per_epoch

        if baseline > best_reward:
            best_reward = baseline
            best_weights = policy.weights.copy()
            best_bias = policy.bias.copy()

    policy.weights = best_weights
    policy.bias = best_bias

    np.savez(
        OUT / "policy_weights.npz",
        weights=policy.weights,
        bias=policy.bias,
        actions=np.array(ACTIONS, dtype=object),
    )
    summary = {
        "resources": {"policy_training_rollouts": rollout_calls, "unique_encounter_states": len(rollout_states), "circuit_shots": 0},
        "seed": seed,
        "epochs": epochs,
        "episodes_per_epoch": episodes_per_epoch,
        "best_mean_reward": round(best_reward, 6),
        "final_mean_reward": reward_trace[-1] if reward_trace else 0.0,
        "reward_trace_tail": reward_trace[-10:],
        "artifact": str(OUT / "policy_weights.npz"),
    }
    (OUT / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description="Train the Phase 1 aviation policy baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--episodes-per-epoch", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.045)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/reproduction/phase1")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    OUT = args.output_dir
    for name in ("policy_weights.npz", "training_summary.json"):
        if (OUT / name).exists() and not args.force:
            parser.error(f"Refusing to overwrite {OUT / name}")

    summary = train_policy(
        seed=args.seed,
        epochs=args.epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        learning_rate=args.learning_rate,
    )
    print("Phase 1 training complete")
    print(f"Best mean reward : {summary['best_mean_reward']}")
    print(f"Artifact         : {summary['artifact']}")


if __name__ == "__main__":
    main()
