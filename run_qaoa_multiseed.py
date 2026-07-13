"""Backend-verified, two-pass QAOA multi-seed provenance runner."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import qiskit
import qiskit_aer

from phase2_qaoa.qaoa_runner import QAOAExplorer, QISKIT_AVAILABLE


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
SEEDS = list(range(42, 52))
K = 50
SHOTS = 1024
REPS = 2
COBYLA_MAXITER = 35


def auc(cumulative: Iterable[int]) -> float:
    previous = 0.0
    total = 0.0
    for current in cumulative:
        total += (previous + float(current)) / 2.0
        previous = float(current)
    return round(total, 6)


def first_failure(cumulative: Sequence[int]) -> int | None:
    return next((index for index, value in enumerate(cumulative, start=1) if int(value) > 0), None)


def mean_pstd(values: Sequence[float]) -> Dict[str, float]:
    return {"mean": round(float(statistics.mean(values)), 6), "std": round(float(statistics.pstdev(values)), 6)}


def verify_qaoa_backend() -> Dict[str, Any]:
    if not QISKIT_AVAILABLE:
        raise RuntimeError("ABORT: QAOA backend fell back to a non-Qiskit implementation")
    explorer = QAOAExplorer(reps=REPS, shots=16, seed=42)
    if "fallback" in explorer.backend_name.lower():
        raise RuntimeError(f"ABORT: QAOA backend fell back: {explorer.backend_name}")
    return {
        "qiskit_version": qiskit.__version__,
        "qiskit_aer_version": qiskit_aer.__version__,
        "backend": explorer.backend_name,
    }


def run_once(seed: int) -> Dict[str, Any]:
    explorer = QAOAExplorer(reps=REPS, shots=SHOTS, seed=seed)
    result = explorer.run(k_iterations=K, output_path=None)
    cumulative = [int(value) for value in result["cumulative_failures"]]
    return {
        "seed": seed,
        "unique_failures": int(result["total_unique_failures"]),
        "auc": auc(cumulative),
        "first_failure": first_failure(cumulative),
        "cumulative_failures": cumulative,
        "total_env_evaluations": int(result["n_env_evaluations_total"]),
        "unique_env_evaluations": int(result["n_env_evaluations_unique"]),
        "elapsed_seconds": float(result["elapsed_seconds"]),
        "declared_optimizer_query_cost": int(K * COBYLA_MAXITER * SHOTS),
        "circuit_info": result["circuit_info"],
    }


def checked_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    return {key: row[key] for key in ("unique_failures", "auc", "first_failure", "cumulative_failures", "total_env_evaluations", "unique_env_evaluations")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QAOA p=2 across seeds 42--51 twice")
    parser.add_argument("--force", action="store_true", help="Allow replacement of qaoa_multiseed.json.")
    parser.add_argument("--seed", type=int, choices=SEEDS, help="Run and persist one duplicate-checked seed part.")
    parser.add_argument("--assemble", action="store_true", help="Assemble the final JSON from all verified per-seed parts.")
    args = parser.parse_args()
    if args.seed is not None and args.assemble:
        parser.error("--seed and --assemble are mutually exclusive")
    output = OUT / "qaoa_multiseed.json"
    backend = verify_qaoa_backend()

    parts_dir = OUT / "qaoa_multiseed_parts"
    if args.seed is not None:
        parts_dir.mkdir(parents=True, exist_ok=True)
        part = parts_dir / f"qaoa_seed_{args.seed}.json"
        if part.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite verified seed part: {part}")
        seed = args.seed
        first = run_once(seed)
        second = run_once(seed)
        deterministic = checked_metrics(first) == checked_metrics(second)
        if not deterministic:
            raise RuntimeError(f"ABORT: non-deterministic QAOA checked metrics for seed {seed}")
        print(f"seed={seed} failures={first['unique_failures']} auc={first['auc']} env={first['total_env_evaluations']}/{first['unique_env_evaluations']} deterministic={deterministic}")
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "backend": backend,
            "config": {"p": REPS, "shots": SHOTS, "k": K, "cobyla_maxiter": COBYLA_MAXITER, "protocol": "eight most-frequent bitstrings per iteration"},
            "deterministic": deterministic,
            "record": first,
        }
        part.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {part}")
        return

    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing canonical output: {output}")
    if not args.assemble:
        parser.error("Use --seed SEED for a bounded run, then --assemble after all ten parts exist")
    records: list[Dict[str, Any]] = []
    determinism: Dict[str, bool] = {}
    producing_commit: str | None = None
    for seed in SEEDS:
        part = parts_dir / f"qaoa_seed_{seed}.json"
        if not part.exists():
            raise FileNotFoundError(f"Missing verified seed part: {part}")
        data = json.loads(part.read_text(encoding="utf-8"))
        if producing_commit is None:
            producing_commit = str(data["code_commit"])
        elif data["code_commit"] != producing_commit:
            raise RuntimeError(f"ABORT: seed part from different producing commit: {part}")
        if not data.get("deterministic"):
            raise RuntimeError(f"ABORT: non-deterministic seed part: {part}")
        records.append(data["record"])
        determinism[str(seed)] = True
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": producing_commit,
        "assembly_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "backend": backend,
        "config": {"seeds": SEEDS, "p": REPS, "shots": SHOTS, "k": K, "cobyla_maxiter": COBYLA_MAXITER, "protocol": "eight most-frequent bitstrings per iteration"},
        "determinism_checked": determinism,
        "per_seed": records,
        "aggregate": {
            "unique_failures": mean_pstd([float(row["unique_failures"]) for row in records]),
            "auc": mean_pstd([float(row["auc"]) for row in records]),
            "first_failure": mean_pstd([float(row["first_failure"]) for row in records if row["first_failure"] is not None]),
            "total_env_evaluations": mean_pstd([float(row["total_env_evaluations"]) for row in records]),
            "unique_env_evaluations": mean_pstd([float(row["unique_env_evaluations"]) for row in records]),
            "elapsed_seconds": mean_pstd([float(row["elapsed_seconds"]) for row in records]),
            "declared_optimizer_query_cost": K * COBYLA_MAXITER * SHOTS,
        },
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
