"""
QAA Phase 2 - Monte Carlo baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase2_qaoa.qaoa_runner import OUT, evaluate_state
from phase2_qaoa.qubo_encoder import PARAM_GRID


def random_state(rng: np.random.Generator) -> Dict[str, float]:
    return {
        name: float(values[int(rng.integers(0, len(values)))])
        for name, values in PARAM_GRID.items()
    }


def run_monte_carlo(k_iterations: int = 50, seed: int = 42, output_path: str | Path | None = OUT / "mc_results.json") -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    started = time.time()
    unique_failures: List[Dict[str, Any]] = []
    seen: set[str] = set()
    cumulative: List[int] = []

    for iteration in range(k_iterations):
        evaluation = evaluate_state(random_state(rng))
        evaluation["iteration"] = iteration
        if evaluation["failure"]:
            key = json.dumps(evaluation["params"], sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique_failures.append(evaluation)
        cumulative.append(len(unique_failures))

    results = {
        "method": "MonteCarlo",
        "seed": seed,
        "k_iterations": k_iterations,
        "elapsed_seconds": round(time.time() - started, 3),
        "total_unique_failures": len(unique_failures),
        "cumulative_failures": cumulative,
        "failures": unique_failures,
    }
    if output_path is not None:
        path = Path(output_path)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="QAA Phase 2 - Monte Carlo baseline")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = run_monte_carlo(k_iterations=args.k, seed=args.seed)
    print(f"Monte Carlo failures: {results['total_unique_failures']}")
    print(f"Results written to   : {OUT / 'mc_results.json'}")


if __name__ == "__main__":
    main()
