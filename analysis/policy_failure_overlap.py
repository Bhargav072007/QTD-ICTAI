"""Overlap between fixed-ego (geometric) failures, policy-conditioned failures,
and the student warm-start set.  Added for the camera-ready.

Requires outputs/phase1/policy_weights.npz (python phase1/train.py regenerates it
deterministically).  Output: outputs/policy_failure_overlap.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phase2_qaoa.qaoa_runner import evaluate_state  # noqa: E402
from phase2_qaoa.qubo_encoder import enumerate_parameter_states  # noqa: E402
from phase3_quantum_tree.pipeline import evaluate_policy_state  # noqa: E402


def key(p):
    return (p["int_heading"], p["int_altitude"], p["int_speed"], p["int_x_offset"])


def main() -> None:
    states = enumerate_parameter_states()
    geo = {key(s) for s in states if evaluate_state(s)["failure"]}
    pol, actions = set(), Counter()
    for s in states:
        r = evaluate_policy_state(s)
        actions[r["action"]] += 1
        if r["failure"]:
            pol.add(key(s))
    rows = json.loads((ROOT / "outputs" / "failure_states.json").read_text())["failure_states"]
    warm = {key(r["params"]) for r in rows}
    out = {
        "geometric_failures": len(geo),
        "policy_failures": len(pol),
        "policy_failures_also_geometric": len(pol & geo),
        "policy_failures_safe_under_fixed_ego": len(pol - geo),
        "geometric_failures_avoided_by_policy": len(geo - pol),
        "policy_primary_action_counts_over_256_states": dict(actions),
        "warmstart_records": len(rows),
        "warmstart_unique_states": len(warm),
        "warmstart_equals_policy_failure_set": warm == pol,
        "warmstart_states_that_are_geometric_failures": len(warm & geo),
        "policy_failure_states": sorted(pol),
    }
    path = ROOT / "outputs" / "policy_failure_overlap.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print({k: v for k, v in out.items() if k != "policy_failure_states"})


if __name__ == "__main__":
    main()
