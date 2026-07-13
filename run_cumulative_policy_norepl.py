"""Generate no-replacement cumulative MC curves and policy-loop MC rerun."""

from __future__ import annotations

import json
import argparse
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import qiskit
import qiskit_aer

from phase2_qaoa.qaoa_runner import evaluate_state
from phase2_qaoa.qubo_encoder import enumerate_parameter_states
from phase3_quantum_tree.pipeline import evaluate_policy_state
from phase3_quantum_tree.quantum_layer import QuantumDistillationLayer


ROOT = Path(__file__).resolve().parent
FIG_OUT = ROOT / "outputs" / "fig_cumulative_norepl.json"
POLICY_OUT = ROOT / "outputs" / "policy_loop_norepl.json"
SEEDS = list(range(42, 52))
K = 50

State = Dict[str, float]
StateKey = Tuple[float, float, float, float]


def state_key(state: State) -> StateKey:
    return (
        float(state["int_heading"]),
        float(state["int_altitude"]),
        float(state["int_speed"]),
        float(state["int_x_offset"]),
    )


def auc(values: Iterable[int]) -> float:
    total = 0.0
    previous = 0.0
    for current in values:
        total += (previous + float(current)) / 2.0
        previous = float(current)
    return round(total, 6)


def first_failure(cumulative: Sequence[int]) -> int | None:
    for index, value in enumerate(cumulative, start=1):
        if int(value) > 0:
            return index
    return None


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": round(float(statistics.mean(values)), 6),
        "std": round(float(statistics.pstdev(values)), 6),
    }


def format_mean_std(stats: Dict[str, float]) -> str:
    return f"{stats['mean']}+/-{stats['std']}"


def verify_quantum_backend() -> Dict[str, Any]:
    layer = QuantumDistillationLayer(shots=16, seed=42)
    out = layer.refine(np.ones((4, 8), dtype=float), np.array([0.9, 0.8, 0.7, 0.6], dtype=float))
    backend_values = sorted(set(str(value) for value in out["backend"]))
    if "qiskit-statevector" not in backend_values:
        raise RuntimeError(f"ABORT: quantum backend fell back; backend_values={backend_values}")
    return {
        "qiskit_version": qiskit.__version__,
        "qiskit_aer_version": qiskit_aer.__version__,
        "quantum_backend_values": backend_values,
    }


def mc_without_replacement(
    seed: int,
    states: Sequence[State],
    evaluator,
    *,
    k: int = K,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    draw_count = min(k, len(states))
    indices = rng.choice(len(states), size=draw_count, replace=False)
    found: set[StateKey] = set()
    cumulative: List[int] = []
    for index in indices:
        state = states[int(index)]
        result = evaluator(state)
        if result["failure"]:
            found.add(state_key(state))
        cumulative.append(len(found))
    return {
        "seed": seed,
        "k": draw_count,
        "sampling": "uniform without replacement",
        "unique_failures": len(found),
        "auc": auc(cumulative),
        "first_failure_iteration": first_failure(cumulative),
        "cumulative_failures": cumulative,
    }


def add_row(rows: List[Dict[str, Any]], paper_number: str, new_value: Any, old_value: Any) -> None:
    rows.append(
        {
            "paper_number": paper_number,
            "new_value": new_value,
            "old_value": old_value,
            "changed": str(new_value) != str(old_value),
        }
    )


def reproduction_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "outputs" / "reproduction" / f"cumulative_policy_norepl_{stamp}"


def resolve_outputs(output_dir: Path | None, force: bool) -> tuple[Path, Path]:
    if output_dir is None:
        fig_out = FIG_OUT
        policy_out = POLICY_OUT
        if (fig_out.exists() or policy_out.exists()) and not force:
            base = reproduction_dir()
            fig_out = base / FIG_OUT.name
            policy_out = base / POLICY_OUT.name
    else:
        fig_out = output_dir / FIG_OUT.name
        policy_out = output_dir / POLICY_OUT.name
    for path in (fig_out, policy_out):
        if path.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    return fig_out, policy_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate no-replacement cumulative and policy-loop MC outputs")
    parser.add_argument("--output-dir", type=Path, default=None, help="Write outputs here instead of the canonical outputs directory.")
    parser.add_argument("--force", action="store_true", help="Allow overwriting the selected output files.")
    args = parser.parse_args()
    fig_out, policy_out = resolve_outputs(args.output_dir, args.force)

    backend = verify_quantum_backend()
    states = enumerate_parameter_states()
    canonical_fig = json.loads((ROOT / "outputs" / "fig_cumulative_data.json").read_text(encoding="utf-8"))
    policy_old = json.loads((ROOT / "outputs" / "policy_loop_comparison.json").read_text(encoding="utf-8"))

    seed42_mc = mc_without_replacement(42, states, evaluate_state, k=K)
    if seed42_mc["unique_failures"] != 3 or seed42_mc["auc"] != 44.5 or seed42_mc["first_failure_iteration"] != 13:
        raise RuntimeError(
            "Seed-42 MC no-replacement did not reproduce expected run: "
            f"found={seed42_mc['unique_failures']} auc={seed42_mc['auc']} "
            f"first_failure={seed42_mc['first_failure_iteration']}"
        )
    qtd_cum42 = [int(value) for value in canonical_fig["qtd"]["cumulative_failures"]]
    if qtd_cum42[-1] != 18:
        raise RuntimeError(f"Canonical QTD cumulative does not end at 18: final={qtd_cum42[-1]}")

    fig_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "config": {"seed": 42, "k": K, "sampling": "uniform without replacement"},
        "mc_cum42": seed42_mc["cumulative_failures"],
        "qtd_cum42": qtd_cum42,
        "mc_seed42_summary": {
            "found": seed42_mc["unique_failures"],
            "denominator": 18,
            "auc": seed42_mc["auc"],
            "first_failure_iteration": seed42_mc["first_failure_iteration"],
        },
        "qtd_seed42_summary": {
            "found": int(canonical_fig["qtd"]["total_unique_failures"]),
            "denominator": 18,
            "auc": float(canonical_fig["qtd"]["auc"]),
        },
    }
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    fig_out.write_text(json.dumps(fig_payload, indent=2), encoding="utf-8")

    reachable_failures = sum(1 for state in states if evaluate_policy_state(state)["failure"])
    if reachable_failures != 12:
        raise RuntimeError(f"Expected 12 reachable policy failures; got {reachable_failures}")
    policy_mc_records = [mc_without_replacement(seed, states, evaluate_policy_state, k=K) for seed in SEEDS]
    policy_mc_aggregate = {
        "unique_failures": mean_std([float(row["unique_failures"]) for row in policy_mc_records]),
        "auc": mean_std([float(row["auc"]) for row in policy_mc_records]),
    }

    old_mc_unique = policy_old["aggregate"]["monte_carlo"]["unique_failures"]
    old_qtd_unique = policy_old["aggregate"]["qtd_quantum_on"]["unique_failures"]
    old_teacher_unique = policy_old["aggregate"]["teacher_only_no_quantum"]["unique_failures"]
    rows: List[Dict[str, Any]] = []
    add_row(
        rows,
        "Experiment 4 closed-loop MC reachable failures mean+/-std",
        format_mean_std(policy_mc_aggregate["unique_failures"]),
        format_mean_std(old_mc_unique),
    )

    policy_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "config": {
            "seeds": SEEDS,
            "k": K,
            "sampling": "uniform without replacement",
            "reachable_failures_denominator": reachable_failures,
        },
        "mc_records": policy_mc_records,
        "mc_aggregate": policy_mc_aggregate,
        "unchanged_comparators": {
            "qtd_quantum_on_unique_failures": old_qtd_unique,
            "teacher_only_no_quantum_unique_failures": old_teacher_unique,
        },
        "closed_loop_mc_paper_table": rows,
    }
    policy_out.write_text(json.dumps(policy_payload, indent=2), encoding="utf-8")

    print("BACKEND CHECK")
    print(f"qiskit_version={backend['qiskit_version']}")
    print(f"qiskit_aer_version={backend['qiskit_aer_version']}")
    print(f"quantum_backend_values={','.join(backend['quantum_backend_values'])}")
    print("")
    print("TASK 1 SEED-42 CUMULATIVE ARRAYS")
    print(f"mc_cum42={json.dumps(seed42_mc['cumulative_failures'])}")
    print(f"qtd_cum42={json.dumps(qtd_cum42)}")
    print(
        "seed42_mc_summary="
        f"found={seed42_mc['unique_failures']}/18 auc={seed42_mc['auc']} "
        f"first_failure_iteration={seed42_mc['first_failure_iteration']}"
    )
    print(f"seed42_qtd_final={qtd_cum42[-1]}")
    print(f"wrote_fig={fig_out}")
    print("")
    print("TASK 2 CLOSED-LOOP POLICY MC WITHOUT REPLACEMENT")
    print(f"reachable_failures_denominator={reachable_failures}")
    print(f"mc_reachable_failures_mean_std={format_mean_std(policy_mc_aggregate['unique_failures'])}")
    print(f"mc_auc_mean_std={format_mean_std(policy_mc_aggregate['auc'])}")
    print(f"qtd_unchanged_reachable_failures_mean_std={format_mean_std(old_qtd_unique)}")
    print(f"teacher_only_unchanged_reachable_failures_mean_std={format_mean_std(old_teacher_unique)}")
    print("paper_number | new_value | old_value | changed")
    for row in rows:
        print(f"{row['paper_number']} | {row['new_value']} | {row['old_value']} | {row['changed']}")
    print(f"wrote_policy={policy_out}")


if __name__ == "__main__":
    main()
