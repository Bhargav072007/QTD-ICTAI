"""
Phase 1 - evaluate the trained baseline policy and export empirical failure states.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase1.aviation_env_3d import ACTIONS, AviationEnv3D, state_to_indices
from phase1.train import OUT, LinearPolicy, train_policy


OUTPUT_ROOT = ROOT / "outputs"


def _load_policy(seed: int = 42, train_if_missing: bool = True) -> LinearPolicy:
    artifact = OUT / "policy_weights.npz"
    if not artifact.exists():
        if not train_if_missing:
            raise FileNotFoundError(f"Missing policy artifact at {artifact}")
        train_policy(seed=seed)
    data = np.load(artifact, allow_pickle=True)
    policy = LinearPolicy(seed=seed)
    policy.weights = np.array(data["weights"], dtype=float)
    policy.bias = np.array(data["bias"], dtype=float)
    return policy


def evaluate_policy(episodes: int = 500, seed: int = 42) -> Dict[str, Any]:
    env = AviationEnv3D(seed=seed)
    policy = _load_policy(seed=seed)
    rewards: List[float] = []
    rollout_states = set()
    failure_states: List[Dict[str, Any]] = []
    success_count = 0
    failure_counts = {"separation_loss": 0, "near_miss": 0}

    for episode in range(episodes):
        obs = env.reset()
        probs = policy.action_probs(obs)
        # Greedy action selection (argmax). Policy always picks highest-prob action.
        # All 500 episodes will deterministically follow the same action for each state.
        action_index = int(np.argmax(probs))
        _, reward, _, outcome = env.step(action_index)
        rewards.append(reward)
        rollout_states.add(tuple(sorted(outcome["params"].items())))
        if outcome["failure"]:
            failure_counts[outcome["failure_type"]] += 1
            failure_states.append(
                {
                    "episode": episode,
                    "params": outcome["params"],
                    "indices": list(state_to_indices(outcome["params"])),
                    "action": ACTIONS[action_index],
                    "action_index": action_index,
                    "failure_type": outcome["failure_type"],
                    "min_h_sep_nm": outcome["min_h_sep_nm"],
                    "min_v_sep_ft": outcome["min_v_sep_ft"],
                    "reward": outcome["reward"],
                }
            )
        else:
            success_count += 1

    summary = {
        "resources": {"label_acquisition_rollouts": len(rewards), "unique_encounter_states": len(rollout_states), "circuit_shots": 0},
        "episodes": episodes,
        "seed": seed,
        "mean_reward": round(float(np.mean(rewards)), 6),
        "success_rate": round(success_count / max(episodes, 1), 6),
        "failure_rate": round(len(failure_states) / max(episodes, 1), 6),
        "near_miss_rate": round(failure_counts["near_miss"] / max(episodes, 1), 6),
        "separation_loss_rate": round(failure_counts["separation_loss"] / max(episodes, 1), 6),
        "failure_counts": failure_counts,
        # Record the actual policy location selected for this reproduction.
        "policy_artifact": str(OUT / "policy_weights.npz"),
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "failure_states.json").write_text(
        json.dumps({"episodes": episodes, "failure_states": failure_states}, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_ROOT / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    global OUT, OUTPUT_ROOT
    parser = argparse.ArgumentParser(description="Evaluate the Phase 1 aviation policy baseline")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/reproduction/phase1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--policy-dir", type=Path, default=ROOT / "outputs/reproduction/phase1")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    OUT = args.policy_dir
    OUTPUT_ROOT = args.output_dir
    if not (OUT / "policy_weights.npz").exists():
        parser.error("Train first or select --policy-dir containing policy_weights.npz")
    for path in (OUTPUT_ROOT / "failure_states.json", OUTPUT_ROOT / "evaluation_summary.json"):
        if path.exists() and not args.force:
            parser.error(f"Refusing to overwrite {path}")

    summary = evaluate_policy(episodes=args.episodes, seed=args.seed)
    print("Phase 1 evaluation complete")
    print(f"Episodes            : {summary['episodes']}")
    print(f"Mean reward         : {summary['mean_reward']}")
    print(f"Failure rate        : {summary['failure_rate']}")
    print(f"Separation loss rate: {summary['separation_loss_rate']}")
    print(f"Near miss rate      : {summary['near_miss_rate']}")
    print(f"Failure states      : {OUTPUT_ROOT / 'failure_states.json'}")


if __name__ == "__main__":
    main()
