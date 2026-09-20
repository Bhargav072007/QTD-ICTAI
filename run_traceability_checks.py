"""
Traceability checks for the CURRENT QTD ICTAI-2026 manuscript claims.

Unlike the earlier version (which re-ran experiments and checked a stale draft:
336.0 AUC, +104%, student MSE), this rebuild READS the persisted result JSONs,
recomputes AUCs with the shared trapezoidal metric, and checks each CURRENT
paper claim against its named JSON source. Every row is expected to be MATCH or
ROUNDING-MATCH; anything else is a FAIL that must be reconciled before submission.

It also (a) writes outputs/stats_significance.json with exact two-sided sign
tests and a Wilcoxon signed-rank test, and (b) rewrites
outputs/paper_metric_traceability.json with the current-claim table.

Reproduce:
    python run_traceability_checks.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from scipy.stats import binomtest, wilcoxon

from phase3_quantum_tree.classical_layer import run_classical_layer


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def load(name: str) -> Dict[str, Any]:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def auc(cumulative: Sequence[int]) -> float:
    """Shared trapezoidal cumulative-failure AUC starting from previous=0."""
    total = 0.0
    previous = 0.0
    for current in cumulative:
        total += (previous + float(current)) / 2.0
        previous = float(current)
    return round(total, 3)


def time_to_first_failure(cumulative: Sequence[int]) -> Optional[int]:
    for index, value in enumerate(cumulative, start=1):
        if int(value) > 0:
            return index
    return None


def _round_half_up(value: float, decimals: int) -> float:
    quant = Decimal(1) if decimals <= 0 else Decimal("1." + "0" * decimals)
    return float(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))


def classify(paper: Any, persisted: Any, decimals: int) -> str:
    if persisted is None:
        return "FAIL"
    if isinstance(paper, (int, float)) and isinstance(persisted, (int, float)):
        if float(paper) == float(persisted):
            return "MATCH"
        if _round_half_up(float(persisted), decimals) == float(paper):
            return "ROUNDING-MATCH"
        return "FAIL"
    return "MATCH" if paper == persisted else "FAIL"


class Checker:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def check(
        self,
        name: str,
        paper: Any,
        persisted: Any,
        decimals: int,
        source: str,
        key_path: str,
        notes: str = "",
    ) -> None:
        status = classify(paper, persisted, decimals)
        self.rows.append(
            {
                "metric_name": name,
                "paper_value": paper,
                "persisted_value": persisted,
                "match_status": status,
                "source_json_file": source,
                "source_json_key_path": key_path,
                "notes": notes,
            }
        )

    def check_mean_std(
        self,
        name: str,
        paper_mean: float,
        paper_std: float,
        agg: Dict[str, Any],
        dp_mean: int,
        dp_std: int,
        source: str,
        key_path: str,
        notes: str = "",
    ) -> None:
        self.check(f"{name} (mean)", paper_mean, agg.get("mean"), dp_mean, source, f"{key_path}.mean", notes)
        self.check(f"{name} (std)", paper_std, agg.get("std"), dp_std, source, f"{key_path}.std", notes)


def compute_significance(ablation: Dict[str, Any], split: Dict[str, Any]) -> Dict[str, Any]:
    ablation_lifts = [float(row["delta"]["auc"]) for row in ablation["per_seed"]]
    split_lifts = [
        round(float(row["qtd_auc"]) - float(row["teacher_only_auc"]), 6)
        for row in split["per_split"]
    ]

    def sign_and_wilcoxon(lifts: Sequence[float]) -> Dict[str, Any]:
        non_zero = [v for v in lifts if v != 0.0]
        positive = sum(1 for v in non_zero if v > 0)
        sign_p = binomtest(positive, len(non_zero), p=0.5, alternative="two-sided").pvalue
        try:
            wil_p = float(
                wilcoxon(non_zero, alternative="two-sided", method="exact", zero_method="wilcox").pvalue
            )
        except ValueError:
            wil_p = None
        return {
            "lifts": list(lifts),
            "n": len(lifts),
            "n_nonzero": len(non_zero),
            "positive_count": int(positive),
            "negative_count": int(len(non_zero) - positive),
            "zero_count": int(len(lifts) - len(non_zero)),
            "sign_test_two_sided_p": round(float(sign_p), 12),
            "wilcoxon_signed_rank_two_sided_p": None if wil_p is None else round(wil_p, 12),
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Exact two-sided sign tests and Wilcoxon signed-rank tests on the per-seed "
            "quantum AUC lifts. Ablation lift is the in-sample QTD-minus-teacher-only AUC "
            "delta (quantum_ablation_multiseed.json); split lift is the current 1024-shot "
            "held-out delta (split_results_1024.json)."
        ),
        "ablation_auc_lift": {
            "source": "outputs/quantum_ablation_multiseed.json#/per_seed[*]/delta/auc",
            **sign_and_wilcoxon(ablation_lifts),
            "expected_sign_test_p_10_of_10": round(2.0 / (2 ** 10), 12),
        },
        "split_auc_lift": {
            "source": "outputs/split_results_1024.json#/per_split[*]/(qtd_auc-teacher_only_auc)",
            **sign_and_wilcoxon(split_lifts),
        },
    }


def main() -> int:
    mc = load("mc_no_replacement.json")
    multiseed = load("multiseed_results.json")
    ablation = load("quantum_ablation_multiseed.json")
    policy = load("policy_loop_comparison.json")
    policy_norepl = load("policy_loop_norepl.json")
    split = load("split_results.json")
    split_1024 = load("split_results_1024.json")
    mechanism = load("mechanism_controls.json")
    adaptive = load("adaptive_baseline.json")
    adaptive_policy = load("adaptive_baseline_policy.json")
    coldstart = load("coldstart_qtd.json")
    qubo = load("qubo_alignment_final.json")
    sweep = load("quantum_8dim_sweep.json")
    phase1 = load("phase1/evaluation_summary.json")
    headline = load("headline_reproducibility_seed42.json")
    ranking = load("ranking_diagnostic_seed42.json")
    qaoa = load("qaoa_results_k50.json")
    qaoa_multiseed = load("qaoa_multiseed.json")
    # Teacher accuracy/loss are in-sample over the full 256-state grid and fully
    # deterministic; recompute them here rather than depend on a persisted fair-run.
    teacher_metrics = run_classical_layer(seed=42)["teacher_metrics"]

    # ---- Significance stats (written first so the table can reference them) ----
    significance = compute_significance(ablation, split_1024)
    (OUT / "stats_significance.json").write_text(json.dumps(significance, indent=2), encoding="utf-8")

    checker = Checker()

    # ---- Table I (seed 42): recompute AUCs with the shared trapezoid ----
    qtd_cum = headline["canonical_current_code"]["cumulative_failures"]
    checker.check("Table I QTD found/18", 18, int(headline["canonical_current_code"]["unique_failures"]), 0, "headline_reproducibility_seed42.json", "canonical_current_code.unique_failures")
    checker.check("Table I QTD AUC", 366.0, auc(qtd_cum), 1, "headline_reproducibility_seed42.json", "canonical_current_code.cumulative_failures")
    checker.check("Table I QTD first-failure iter", 17, time_to_first_failure(qtd_cum), 0, "headline_reproducibility_seed42.json", "canonical_current_code.cumulative_failures")

    to_cum = ranking["teacher_only"]["cumulative_failures"]
    checker.check("Table I teacher-only found/18", 16, int(ranking["teacher_only"]["total_unique_failures"]), 0, "ranking_diagnostic_seed42.json", "teacher_only.total_unique_failures")
    checker.check("Table I teacher-only AUC", 313.0, auc(to_cum), 1, "ranking_diagnostic_seed42.json", "teacher_only.cumulative_failures")
    checker.check("Table I teacher-only first-failure iter", 17, time_to_first_failure(to_cum), 0, "ranking_diagnostic_seed42.json", "teacher_only.cumulative_failures")

    mc_cum = mc["table_i_seed42_mc"]["cumulative_failures"]
    checker.check("Table I MC found/18", 3, int(mc["table_i_seed42_mc"]["unique_failures"]), 0, "mc_no_replacement.json", "table_i_seed42_mc.unique_failures")
    checker.check("Table I MC AUC", 44.5, auc(mc_cum), 1, "mc_no_replacement.json", "table_i_seed42_mc.cumulative_failures")
    checker.check("Table I MC first-failure iter", 13, time_to_first_failure(mc_cum), 0, "mc_no_replacement.json", "table_i_seed42_mc.cumulative_failures")

    qaoa_cum = qaoa["cumulative_failures"]
    qaoa_unique = int(qaoa.get("unique_failures", qaoa.get("total_unique_failures")))
    checker.check("Table I QAOA found/18", 5, qaoa_unique, 0, "qaoa_results_k50.json", "unique_failures|total_unique_failures")
    checker.check("Table I QAOA AUC", 172.5, auc(qaoa_cum), 1, "qaoa_results_k50.json", "cumulative_failures")
    checker.check("Table I QAOA first-failure iter", 4, time_to_first_failure(qaoa_cum), 0, "qaoa_results_k50.json", "cumulative_failures")
    checker.check("QAOA env evaluations total <= 400", True, int(qaoa["n_env_evaluations_total"]) <= 400, 0, "qaoa_results_k50.json", "n_env_evaluations_total", f"total={qaoa['n_env_evaluations_total']} unique={qaoa['n_env_evaluations_unique']}")
    checker.check_mean_std("QAOA multi-seed failures (requires paper delta)", 3.5, 0.8, qaoa_multiseed["aggregate"]["unique_failures"], 1, 1, "qaoa_multiseed.json", "aggregate.unique_failures")
    checker.check_mean_std("QAOA multi-seed AUC (requires paper delta)", 133.9, 27.9, qaoa_multiseed["aggregate"]["auc"], 1, 1, "qaoa_multiseed.json", "aggregate.auc")
    checker.check("QAOA multi-seed completed seeds", 10, len(qaoa_multiseed["per_seed"]), 0, "qaoa_multiseed.json", "per_seed")
    checker.check("QAOA multi-seed duplicate checks", True, all(qaoa_multiseed["determinism_checked"].values()), 0, "qaoa_multiseed.json", "determinism_checked")

    # ---- Table II (10 seeds) ----
    checker.check_mean_std("Table II QTD failing states", 17.9, 0.3, multiseed["aggregate"]["qtd"]["unique_failures"], 1, 1, "multiseed_results.json", "aggregate.qtd.unique_failures")
    checker.check_mean_std("Table II QTD AUC", 378.2, 20.0, multiseed["aggregate"]["qtd"]["auc"], 1, 1, "multiseed_results.json", "aggregate.qtd.auc")
    checker.check_mean_std("Table II QTD first-failure", 15.1, 1.6, multiseed["aggregate"]["qtd"]["time_to_first_failure"], 1, 1, "multiseed_results.json", "aggregate.qtd.time_to_first_failure")
    checker.check_mean_std("Table II MC failing states", 4.2, 1.1, mc["table_ii_mc_aggregate"]["unique_failures"], 1, 1, "mc_no_replacement.json", "table_ii_mc_aggregate.unique_failures")
    checker.check_mean_std("Table II MC AUC", 98.8, 32.4, mc["table_ii_mc_aggregate"]["auc"], 1, 1, "mc_no_replacement.json", "table_ii_mc_aggregate.auc")
    checker.check_mean_std("Table II MC first-failure", 12.0, 6.9, mc["table_ii_mc_aggregate"]["time_to_first_failure"], 1, 1, "mc_no_replacement.json", "table_ii_mc_aggregate.time_to_first_failure")

    # ---- Ablation ----
    delta = ablation["delta_aggregate"]
    checker.check_mean_std("Ablation AUC lift", 26.5, 12.2, delta["auc"], 1, 1, "quantum_ablation_multiseed.json", "delta_aggregate.auc")
    checker.check("Ablation AUC lift positive seeds", 10, int(delta["positive_auc_delta_seed_count"]), 0, "quantum_ablation_multiseed.json", "delta_aggregate.positive_auc_delta_seed_count")
    checker.check("Ablation AUC lift sign-test p", 0.00195, significance["ablation_auc_lift"]["sign_test_two_sided_p"], 5, "stats_significance.json", "ablation_auc_lift.sign_test_two_sided_p")
    checker.check_mean_std("Ablation failure lift", 1.3, 1.2, delta["unique_failures"], 1, 1, "quantum_ablation_multiseed.json", "delta_aggregate.unique_failures")
    checker.check("Ablation failure lift positive seeds", 6, int(delta["positive_failure_delta_seed_count"]), 0, "quantum_ablation_multiseed.json", "delta_aggregate.positive_failure_delta_seed_count")
    tied_zero = sum(1 for row in ablation["per_seed"] if float(row["delta"]["unique_failures"]) == 0.0)
    checker.check("Ablation failure lift tied seeds", 4, tied_zero, 0, "quantum_ablation_multiseed.json", "per_seed[*].delta.unique_failures==0")

    # ---- Closed loop (policy evaluator, denominator 12) ----
    checker.check("Closed-loop denominator", 12, int(policy_norepl["config"]["reachable_failures_denominator"]), 0, "policy_loop_norepl.json", "config.reachable_failures_denominator")
    checker.check_mean_std("Closed-loop QTD failing states", 12.0, 0.0, policy["aggregate"]["qtd_quantum_on"]["unique_failures"], 1, 1, "policy_loop_comparison.json", "aggregate.qtd_quantum_on.unique_failures")
    checker.check_mean_std("Closed-loop teacher-only failing states", 10.9, 1.2, policy["aggregate"]["teacher_only_no_quantum"]["unique_failures"], 1, 1, "policy_loop_comparison.json", "aggregate.teacher_only_no_quantum.unique_failures")
    checker.check("Closed-loop MC failing states (mean)", 2.2, policy["aggregate"]["monte_carlo"]["unique_failures"]["mean"], 1, "policy_loop_comparison.json", "aggregate.monte_carlo.unique_failures.mean")

    # ---- Legacy 256-shot held-out result (retained for provenance; not a current paper claim) ----
    checker.check_mean_std("Split QTD AUC", 302.3, 23.9, split["aggregate"]["qtd_auc"], 1, 1, "split_results.json", "aggregate.qtd_auc")
    checker.check_mean_std("Split QTD recall", 1.00, 0.00, split["aggregate"]["qtd_recall"], 2, 2, "split_results.json", "aggregate.qtd_recall")
    checker.check_mean_std("Split quantum AUC lift", 7.3, 14.4, split["aggregate"]["quantum_auc_lift"], 1, 1, "split_results.json", "aggregate.quantum_auc_lift")
    checker.check("Split lift sign-test p (6/10)", 0.754, significance["split_auc_lift"]["sign_test_two_sided_p"], 3, "stats_significance.json", "split_auc_lift.sign_test_two_sided_p")
    checker.check_mean_std("Exp5 MC AUC", 73.0, 53.2, mc["experiment_5_mc_aggregate"]["auc"], 1, 1, "mc_no_replacement.json", "experiment_5_mc_aggregate.auc")
    checker.check_mean_std("Exp5 MC recall", 0.356, 0.232, mc["experiment_5_mc_aggregate"]["recall"], 3, 3, "mc_no_replacement.json", "experiment_5_mc_aggregate.recall")

    # ---- Current 1024-shot Experiment 5 paper claims ----
    checker.check_mean_std("Exp5 1024-shot QTD AUC", 302.0, 24.0, split_1024["aggregate"]["qtd_auc"], 0, 0, "split_results_1024.json", "aggregate.qtd_auc")
    checker.check_mean_std("Exp5 1024-shot QTD recall", 1.00, 0.00, split_1024["aggregate"]["qtd_recall"], 2, 2, "split_results_1024.json", "aggregate.qtd_recall")
    checker.check_mean_std("Exp5 1024-shot quantum AUC lift", 7.4, 14.4, split_1024["aggregate"]["quantum_auc_lift"], 1, 1, "split_results_1024.json", "aggregate.quantum_auc_lift")

    # ---- Mechanism controls: these rows establish that the original ablation is not causal evidence ----
    full_vs_teacher = mechanism["paired_comparisons"]["qtd_full"]["vs_teacher_only"]["auc_delta"]["aggregate"]
    checker.check_mean_std("Mechanism full-vs-teacher AUC lift", 26.5, 12.2, full_vs_teacher, 1, 1, "mechanism_controls.json", "paired_comparisons.qtd_full.vs_teacher_only.auc_delta.aggregate")
    for arm in ("no_cz", "analytic", "matched_classical", "shuffled", "random"):
        checker.check(
            f"Mechanism {arm} control recorded",
            True,
            arm in mechanism["arms"],
            0,
            "mechanism_controls.json",
            f"arms.{arm}",
            "A control arm exists and must be considered before attributing the full-arm lift to quantum computation.",
        )

    # ---- Experiments 6--7: recompute from persisted per-seed records ----
    checker.check_mean_std("Experiment 6 CEM failures", 6.5, 2.2, adaptive["aggregate"]["unique_failures"], 1, 1, "adaptive_baseline.json", "aggregate.unique_failures")
    checker.check_mean_std("Experiment 6 CEM AUC", 138.7, 48.0, adaptive["aggregate"]["auc"], 1, 1, "adaptive_baseline.json", "aggregate.auc")
    checker.check_mean_std("Experiment 6 policy CEM failures", 5.2, 1.5, adaptive_policy["aggregate"]["unique_failures"], 1, 1, "adaptive_baseline_policy.json", "aggregate.unique_failures")
    cold = coldstart["aggregate"]
    checker.check_mean_std("Experiment 7 cold-start QTD failures", 3.7, 4.3, cold["coldstart_qtd"]["unique_failures"], 1, 1, "coldstart_qtd.json", "aggregate.coldstart_qtd.unique_failures")
    checker.check_mean_std("Experiment 7 teacher-only failures", 6.1, 2.5, cold["coldstart_teacher_only"]["unique_failures"], 1, 1, "coldstart_qtd.json", "aggregate.coldstart_teacher_only.unique_failures")
    checker.check_mean_std("Experiment 7 failure delta", -2.4, 2.7, cold["quantum_delta"]["unique_failures"], 1, 1, "coldstart_qtd.json", "aggregate.quantum_delta.unique_failures")
    checker.check_mean_std("Experiment 7 AUC delta", -29.3, 44.1, cold["quantum_delta"]["auc"], 1, 1, "coldstart_qtd.json", "aggregate.quantum_delta.auc")
    checker.check("Experiment 7 positive failure deltas", 2, int(cold["quantum_delta"]["positive_failure_delta_seed_count"]), 0, "coldstart_qtd.json", "aggregate.quantum_delta.positive_failure_delta_seed_count")
    checker.check("Experiment 7 positive AUC deltas", 3, int(cold["quantum_delta"]["positive_auc_delta_seed_count"]), 0, "coldstart_qtd.json", "aggregate.quantum_delta.positive_auc_delta_seed_count")

    # ---- QUBO alignment ----
    geo = qubo["metrics"]["empirical_build_qubo_matrix"]["geometric_labels"]
    checker.check("QUBO ranking AUC", 0.753, geo["ranking_auc_score_neg_objective"], 3, "qubo_alignment_final.json", "metrics.empirical_build_qubo_matrix.geometric_labels.ranking_auc_score_neg_objective")
    checker.check("QUBO point-biserial r", -0.191, geo["point_biserial_r_label_vs_objective"], 3, "qubo_alignment_final.json", "metrics.empirical_build_qubo_matrix.geometric_labels.point_biserial_r_label_vs_objective")

    # ---- 8-qubit sweep (36 variants; best-variant mean AUC lift) ----
    best_variant = sweep["top_by_mean_auc_lift"][0]
    best_agg = sweep["aggregate"][best_variant]
    checker.check("8q sweep variant count", 36, len(sweep["variants"]), 0, "quantum_8dim_sweep.json", "variants")
    checker.check("8q sweep best-variant AUC lift (mean)", 7.3, best_agg["auc_lift_vs_teacher"]["mean"], 1, "quantum_8dim_sweep.json", f"aggregate.{best_variant}.auc_lift_vs_teacher.mean", f"best_variant={best_variant}; positive_lift_splits={best_agg['positive_lift_splits']}/10 (not significant)")

    # ---- Teacher (in-sample) + Phase 1 ----
    checker.check("Teacher accuracy", 0.926, teacher_metrics["accuracy"], 3, "phase3_quantum_tree/classical_layer.py", "run_classical_layer(seed=42) in-sample")
    checker.check("Teacher loss", 0.182, teacher_metrics["loss"], 3, "phase3_quantum_tree/classical_layer.py", "run_classical_layer(seed=42) in-sample")
    checker.check("Phase 1 episodes", 500, int(phase1["episodes"]), 0, "phase1/evaluation_summary.json", "episodes")
    checker.check("Phase 1 failure rate (%)", 5.2, float(phase1["failure_rate"]) * 100.0, 1, "phase1/evaluation_summary.json", "failure_rate")
    checker.check("Phase 1 success rate (%)", 94.8, float(phase1["success_rate"]) * 100.0, 1, "phase1/evaluation_summary.json", "success_rate")

    # ---- Persist and report ----
    n_match = sum(1 for r in checker.rows if r["match_status"] == "MATCH")
    n_round = sum(1 for r in checker.rows if r["match_status"] == "ROUNDING-MATCH")
    n_fail = sum(1 for r in checker.rows if r["match_status"] == "FAIL")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "description": "Traceability of the current QTD ICTAI-2026 manuscript claims to persisted JSON sources.",
        "shared_metric_functions": {
            "auc": "trapezoidal cumulative-failure AUC starting from previous=0",
            "time_to_first_failure": "1-based first index where cumulative_failures > 0",
            "std": "population standard deviation (ddof=0)",
        },
        "summary": {
            "total_rows": len(checker.rows),
            "match": n_match,
            "rounding_match": n_round,
            "fail": n_fail,
            "all_pass": n_fail == 0,
        },
        "significance_source": "outputs/stats_significance.json",
        "metrics": checker.rows,
    }
    (OUT / "paper_metric_traceability.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # PASS/FAIL table
    print("PAPER TRACEABILITY (current main.tex claims)")
    print("metric | paper | persisted | status | source")
    print("--- | --- | --- | --- | ---")
    for row in checker.rows:
        print(
            f"{row['metric_name']} | {row['paper_value']} | {row['persisted_value']} | "
            f"{row['match_status']} | {row['source_json_file']}"
        )
    print("")
    print("STATS SIGNIFICANCE")
    ab = significance["ablation_auc_lift"]
    sp = significance["split_auc_lift"]
    print(f"ablation AUC lift: {ab['positive_count']}/{ab['n_nonzero']} positive, sign p={ab['sign_test_two_sided_p']}, wilcoxon p={ab['wilcoxon_signed_rank_two_sided_p']}")
    print(f"split AUC lift:    {sp['positive_count']}/{sp['n_nonzero']} positive, sign p={sp['sign_test_two_sided_p']}, wilcoxon p={sp['wilcoxon_signed_rank_two_sided_p']}")
    print("")
    print(f"SUMMARY: {n_match} MATCH, {n_round} ROUNDING-MATCH, {n_fail} FAIL "
          f"({'ALL PASS' if n_fail == 0 else 'HAS FAILURES'})")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
