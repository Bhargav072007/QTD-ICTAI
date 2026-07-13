"""Budget-fair Monte Carlo rerun and exact ablation significance tests."""

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
from scipy.stats import binomtest, wilcoxon

from phase2_qaoa.qaoa_runner import evaluate_state
from phase2_qaoa.qubo_encoder import enumerate_parameter_states
from phase3_quantum_tree.quantum_layer import QuantumDistillationLayer
from run_split_experiment import state_key, stratified_split


ROOT = Path(__file__).resolve().parent
OUT_JSON = ROOT / "outputs" / "mc_no_replacement.json"
SEEDS = list(range(42, 52))
K = 50
TOTAL_FAILURES = 18


State = Dict[str, float]
StateKey = Tuple[float, float, float, float]


def auc(values: Iterable[int]) -> float:
    total = 0.0
    previous = 0.0
    for current in values:
        total += (previous + float(current)) / 2.0
        previous = float(current)
    return round(total, 6)


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": round(float(statistics.mean(values)), 6),
        "std": round(float(statistics.pstdev(values)), 6),
    }


def first_failure(cumulative: Sequence[int]) -> int | None:
    for index, value in enumerate(cumulative, start=1):
        if int(value) > 0:
            return index
    return None


def labels_for(states: Sequence[State]) -> np.ndarray:
    return np.array([1.0 if evaluate_state(state)["failure"] else 0.0 for state in states], dtype=float)


def mc_without_replacement(seed: int, pool_states: Sequence[State], k: int = K) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    draw_count = min(k, len(pool_states))
    indices = rng.choice(len(pool_states), size=draw_count, replace=False)
    found: set[StateKey] = set()
    cumulative: List[int] = []
    for index in indices:
        state = pool_states[int(index)]
        result = evaluate_state(state)
        if result["failure"]:
            found.add(state_key(state))
        cumulative.append(len(found))
    return {
        "seed": seed,
        "k": draw_count,
        "sampling": "uniform without replacement",
        "unique_failures": len(found),
        "recall": round(len(found) / TOTAL_FAILURES, 6),
        "auc": auc(cumulative),
        "time_to_first_failure": first_failure(cumulative),
        "cumulative_failures": cumulative,
    }


def heldout_mc_without_replacement(seed: int, test_states: Sequence[State], test_labels: np.ndarray) -> Dict[str, Any]:
    result = mc_without_replacement(seed, test_states, k=K)
    test_failures = int(np.sum(test_labels))
    result["test_failures"] = test_failures
    result["recall"] = round(result["unique_failures"] / test_failures, 6) if test_failures else 0.0
    return result


def exact_tests(lifts: Sequence[float]) -> Dict[str, float]:
    non_zero = [float(value) for value in lifts if float(value) != 0.0]
    positive = sum(1 for value in non_zero if value > 0)
    wilcoxon_result = wilcoxon(non_zero, alternative="two-sided", method="exact", zero_method="wilcox")
    sign_result = binomtest(positive, len(non_zero), p=0.5, alternative="two-sided")
    return {
        "wilcoxon_p": round(float(wilcoxon_result.pvalue), 12),
        "sign_test_p": round(float(sign_result.pvalue), 12),
        "positive_count": int(positive),
        "negative_count": int(len(non_zero) - positive),
        "zero_count": int(len(lifts) - len(non_zero)),
    }


def format_mean_std(stats: Dict[str, float]) -> str:
    return f"{stats['mean']}+/-{stats['std']}"


def reproduction_output_path(filename: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "outputs" / "reproduction" / f"mc_no_replacement_{stamp}" / filename


def resolve_output_path(filename: str, output_dir: Path | None, force: bool) -> Path:
    if output_dir is not None:
        path = output_dir / filename
    else:
        path = OUT_JSON if filename == OUT_JSON.name else ROOT / "outputs" / filename
        if path.exists() and not force:
            path = reproduction_output_path(filename)
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    return path


def add_row(rows: List[Dict[str, Any]], number: str, new_value: Any, old_value: Any) -> None:
    rows.append(
        {
            "paper_number": number,
            "new_value": new_value,
            "old_value": old_value,
            "changed": str(new_value) != str(old_value),
        }
    )


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run no-replacement Monte Carlo paper checks")
    parser.add_argument("--output-dir", type=Path, default=None, help="Write outputs here instead of the canonical outputs directory.")
    parser.add_argument("--force", action="store_true", help="Allow overwriting the selected output file.")
    args = parser.parse_args()
    out_json = resolve_output_path(OUT_JSON.name, args.output_dir, args.force)

    backend = verify_quantum_backend()
    states = enumerate_parameter_states()
    labels = labels_for(states)
    failure_count = int(np.sum(labels))
    if len(states) != 256 or failure_count != TOTAL_FAILURES:
        raise RuntimeError(f"Unexpected state/failure count: states={len(states)} failures={failure_count}")

    fig = json.loads((ROOT / "outputs" / "fig_cumulative_data.json").read_text(encoding="utf-8"))
    multiseed_old = json.loads((ROOT / "outputs" / "multiseed_results.json").read_text(encoding="utf-8"))
    ablation = json.loads((ROOT / "outputs" / "quantum_ablation_multiseed.json").read_text(encoding="utf-8"))
    split = json.loads((ROOT / "outputs" / "split_results.json").read_text(encoding="utf-8"))

    seed42_mc = mc_without_replacement(42, states, k=K)
    full_mc_records = [mc_without_replacement(seed, states, k=K) for seed in SEEDS]
    full_firsts = [row["time_to_first_failure"] for row in full_mc_records if row["time_to_first_failure"] is not None]
    full_first_excluded = sum(1 for row in full_mc_records if row["time_to_first_failure"] is None)
    full_mc_aggregate = {
        "unique_failures": mean_std([float(row["unique_failures"]) for row in full_mc_records]),
        "auc": mean_std([float(row["auc"]) for row in full_mc_records]),
        "time_to_first_failure": mean_std([float(value) for value in full_firsts]),
        "first_failure_excluded_zero_failure_seeds": int(full_first_excluded),
    }

    heldout_records: List[Dict[str, Any]] = []
    for seed in SEEDS:
        train_indices, test_indices = stratified_split(states, labels, seed)
        del train_indices
        test_states = [states[index] for index in test_indices]
        test_labels = labels[test_indices]
        heldout_records.append(heldout_mc_without_replacement(seed, test_states, test_labels))
    heldout_mc_aggregate = {
        "recall": mean_std([float(row["recall"]) for row in heldout_records]),
        "auc": mean_std([float(row["auc"]) for row in heldout_records]),
    }

    old_seed42_first = first_failure(fig["monte_carlo"]["cumulative_failures"])
    old_multiseed_mc_records = [row for row in multiseed_old["records"] if row["method"] == "monte_carlo"]
    old_full_first_excluded = sum(1 for row in old_multiseed_mc_records if row.get("time_to_first_failure") is None)

    in_sample_lifts = [
        float(row["qtd_quantum_on"]["auc"]) - float(row["teacher_only_no_quantum"]["auc"])
        for row in ablation["per_seed"]
    ]
    heldout_lifts = [float(row["qtd_auc"]) - float(row["teacher_only_auc"]) for row in split["per_split"]]
    significance = {
        "in_sample_auc_lifts": in_sample_lifts,
        "in_sample": exact_tests(in_sample_lifts),
        "heldout_auc_lifts": heldout_lifts,
        "heldout": exact_tests(heldout_lifts),
    }

    affected_rows: List[Dict[str, Any]] = []
    add_row(affected_rows, "Table I MC found/18", f"{seed42_mc['unique_failures']}/18", f"{fig['monte_carlo']['total_unique_failures']}/18")
    add_row(affected_rows, "Table I MC recall", seed42_mc["recall"], round(float(fig["monte_carlo"]["total_unique_failures"]) / TOTAL_FAILURES, 6))
    add_row(affected_rows, "Table I MC discovery AUC", seed42_mc["auc"], fig["monte_carlo"]["auc"])
    add_row(affected_rows, "Table I MC first-failure iteration", seed42_mc["time_to_first_failure"], old_seed42_first)
    add_row(affected_rows, "Table I QTD found/18", f"{fig['qtd']['total_unique_failures']}/18", f"{fig['qtd']['total_unique_failures']}/18")
    add_row(affected_rows, "Table I QTD discovery AUC", fig["qtd"]["auc"], fig["qtd"]["auc"])
    add_row(
        affected_rows,
        "Table II MC failing states mean+/-std",
        format_mean_std(full_mc_aggregate["unique_failures"]),
        format_mean_std(multiseed_old["aggregate"]["monte_carlo"]["unique_failures"]),
    )
    add_row(
        affected_rows,
        "Table II MC AUC mean+/-std",
        format_mean_std(full_mc_aggregate["auc"]),
        format_mean_std(multiseed_old["aggregate"]["monte_carlo"]["auc"]),
    )
    add_row(
        affected_rows,
        "Table II MC first-failure mean+/-std",
        format_mean_std(full_mc_aggregate["time_to_first_failure"]),
        format_mean_std(multiseed_old["aggregate"]["monte_carlo"]["time_to_first_failure"]),
    )
    add_row(
        affected_rows,
        "Table II MC first-failure excluded seeds",
        full_mc_aggregate["first_failure_excluded_zero_failure_seeds"],
        old_full_first_excluded,
    )
    add_row(
        affected_rows,
        "Table II QTD failing states mean+/-std",
        format_mean_std(multiseed_old["aggregate"]["qtd"]["unique_failures"]),
        format_mean_std(multiseed_old["aggregate"]["qtd"]["unique_failures"]),
    )
    add_row(
        affected_rows,
        "Table II QTD AUC mean+/-std",
        format_mean_std(multiseed_old["aggregate"]["qtd"]["auc"]),
        format_mean_std(multiseed_old["aggregate"]["qtd"]["auc"]),
    )
    add_row(
        affected_rows,
        "Experiment 5 MC recall mean+/-std",
        format_mean_std(heldout_mc_aggregate["recall"]),
        format_mean_std(split["aggregate"]["mc_recall"]),
    )
    add_row(
        affected_rows,
        "Experiment 5 MC discovery AUC mean+/-std",
        format_mean_std(heldout_mc_aggregate["auc"]),
        format_mean_std(split["aggregate"]["mc_auc"]),
    )
    add_row(
        affected_rows,
        "Experiment 5 QTD recall mean+/-std",
        format_mean_std(split["aggregate"]["qtd_recall"]),
        format_mean_std(split["aggregate"]["qtd_recall"]),
    )
    add_row(
        affected_rows,
        "Experiment 5 QTD AUC mean+/-std",
        format_mean_std(split["aggregate"]["qtd_auc"]),
        format_mean_std(split["aggregate"]["qtd_auc"]),
    )
    add_row(
        affected_rows,
        "Experiment 5 teacher-only recall mean+/-std",
        format_mean_std(split["aggregate"]["teacher_only_recall"]),
        format_mean_std(split["aggregate"]["teacher_only_recall"]),
    )
    add_row(
        affected_rows,
        "Experiment 5 teacher-only AUC mean+/-std",
        format_mean_std(split["aggregate"]["teacher_only_auc"]),
        format_mean_std(split["aggregate"]["teacher_only_auc"]),
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "config": {
            "seeds": SEEDS,
            "k": K,
            "sampling": "uniform without replacement",
            "total_states": len(states),
            "total_failures": failure_count,
        },
        "table_i_seed42_mc": seed42_mc,
        "table_ii_mc_records": full_mc_records,
        "table_ii_mc_aggregate": full_mc_aggregate,
        "experiment_5_mc_records": heldout_records,
        "experiment_5_mc_aggregate": heldout_mc_aggregate,
        "unchanged_comparators": {
            "table_i_qtd": fig["qtd"],
            "table_ii_qtd_aggregate": multiseed_old["aggregate"]["qtd"],
            "experiment_5_qtd": {
                "recall": split["aggregate"]["qtd_recall"],
                "auc": split["aggregate"]["qtd_auc"],
            },
            "experiment_5_teacher_only": {
                "recall": split["aggregate"]["teacher_only_recall"],
                "auc": split["aggregate"]["teacher_only_auc"],
            },
        },
        "significance": significance,
        "affected_paper_numbers": affected_rows,
        "notes": (
            "experiment_5_mc_aggregate (recall 0.355556, AUC 73.0) is the CANONICAL Exp-5 "
            "Monte-Carlo source for the paper: uniform WITHOUT replacement over the held-out test "
            "split, exactly 50 unique evaluations, matching the budget convention used for every "
            "other Monte-Carlo row. It supersedes the secondary aggregate.mc_recall/mc_auc figures "
            "embedded in split_results.json (0.266666 / 59.4), which should not be cited for Exp-5."
        ),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("BACKEND CHECK")
    print(f"qiskit_version={backend['qiskit_version']}")
    print(f"qiskit_aer_version={backend['qiskit_aer_version']}")
    print(f"quantum_backend_values={','.join(backend['quantum_backend_values'])}")
    print("")
    print("MC WITHOUT REPLACEMENT RESULTS")
    print(
        "Table I MC seed42: "
        f"found={seed42_mc['unique_failures']}/18 recall={seed42_mc['recall']} "
        f"auc={seed42_mc['auc']} first_failure={seed42_mc['time_to_first_failure']}"
    )
    print(
        "Table II MC: "
        f"failing_states={format_mean_std(full_mc_aggregate['unique_failures'])} "
        f"auc={format_mean_std(full_mc_aggregate['auc'])} "
        f"first_failure={format_mean_std(full_mc_aggregate['time_to_first_failure'])} "
        f"excluded_zero_failure_seeds={full_first_excluded}"
    )
    print(
        "Experiment 5 MC: "
        f"recall={format_mean_std(heldout_mc_aggregate['recall'])} "
        f"auc={format_mean_std(heldout_mc_aggregate['auc'])}"
    )
    print("")
    print("UNCHANGED QTD / TEACHER-ONLY COMPARATORS")
    print(f"Table I QTD: found={fig['qtd']['total_unique_failures']}/18 auc={fig['qtd']['auc']}")
    print(
        "Table II QTD: "
        f"failing_states={format_mean_std(multiseed_old['aggregate']['qtd']['unique_failures'])} "
        f"auc={format_mean_std(multiseed_old['aggregate']['qtd']['auc'])}"
    )
    print(
        "Experiment 5 QTD: "
        f"recall={format_mean_std(split['aggregate']['qtd_recall'])} "
        f"auc={format_mean_std(split['aggregate']['qtd_auc'])}"
    )
    print(
        "Experiment 5 teacher-only: "
        f"recall={format_mean_std(split['aggregate']['teacher_only_recall'])} "
        f"auc={format_mean_std(split['aggregate']['teacher_only_auc'])}"
    )
    print("")
    print("EXACT SIGNIFICANCE")
    print(f"in_sample_auc_lifts={','.join(str(value) for value in in_sample_lifts)}")
    print(f"in_sample_wilcoxon_p={significance['in_sample']['wilcoxon_p']}")
    print(f"in_sample_sign_test_p={significance['in_sample']['sign_test_p']}")
    print(f"heldout_auc_lifts={','.join(str(value) for value in heldout_lifts)}")
    print(f"heldout_wilcoxon_p={significance['heldout']['wilcoxon_p']}")
    print(f"heldout_sign_test_p={significance['heldout']['sign_test_p']}")
    print("")
    print("AFFECTED PAPER NUMBERS")
    print("paper_number | new_value | old_value | changed")
    for row in affected_rows:
        print(f"{row['paper_number']} | {row['new_value']} | {row['old_value']} | {row['changed']}")
    print("")
    print(f"wrote={out_json}")


if __name__ == "__main__":
    main()
