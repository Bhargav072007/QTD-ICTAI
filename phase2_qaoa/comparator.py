"""
QAA Phase 2 - baseline comparators for QAOA, Monte Carlo, and QTRD.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase2_qaoa.qaoa_runner import OUT


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _time_to_first_failure(cumulative: Iterable[int]) -> int | None:
    for index, value in enumerate(cumulative, start=1):
        if value > 0:
            return index
    return None


def _auc(values: Iterable[int]) -> float:
    total = 0.0
    previous = 0.0
    for current in values:
        total += (previous + current) / 2.0
        previous = current
    return round(total, 3)


def _failure_breakdown(failures: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"separation_loss": 0, "near_miss": 0}
    for failure in failures:
        kind = failure.get("failure_type") or failure.get("scenario_type") or failure.get("type") or "near_miss"
        if kind in counts:
            counts[kind] += 1
    return counts


def compare_qtrd_vs_mc() -> Dict[str, Any]:
    qtrd = _load_json(OUT / "quantum_tree_results.json")
    mc = _load_json(OUT / "mc_results.json")

    qtrd_auc = _auc(qtrd["cumulative_failures"])
    mc_auc = _auc(mc["cumulative_failures"])
    qtrd_ttff = _time_to_first_failure(qtrd["cumulative_failures"])
    mc_ttff = _time_to_first_failure(mc["cumulative_failures"])

    comparison = {
        "question": "Can Quantum Tree distillation surface unsafe traffic separation behaviors faster than Monte Carlo?",
        "k_iterations": qtrd["k_iterations"],
        "per_iteration": {
            "qtrd": qtrd["cumulative_failures"],
            "monte_carlo": mc["cumulative_failures"],
        },
        "summary": {
            "qtrd_unique_failures": qtrd["total_unique_failures"],
            "mc_unique_failures": mc["total_unique_failures"],
            "qtrd_auc": qtrd_auc,
            "mc_auc": mc_auc,
            "auc_advantage_pct": round(((qtrd_auc - mc_auc) / mc_auc) * 100.0, 2) if mc_auc else 0.0,
            "qtrd_time_to_first_failure": qtrd_ttff,
            "mc_time_to_first_failure": mc_ttff,
            "qtrd_failure_types": _failure_breakdown(qtrd["top_failures"]),
            "mc_failure_types": _failure_breakdown(mc["failures"]),
            "teacher_accuracy": qtrd["teacher_accuracy"],
            "student_mse": qtrd["student_mse"],
            "quantum_backend": qtrd.get("summary", {}).get("quantum_backend", qtrd["quantum_backend"]),
        },
        "winner": "QTRD" if qtrd["total_unique_failures"] >= mc["total_unique_failures"] else "MonteCarlo",
        "top_failures": qtrd["top_failures"][:10],
    }

    (OUT / "qtrd_vs_mc_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison


def compare_results() -> Dict[str, Any]:
    qaoa = _load_json(OUT / "qaoa_results.json")
    mc = _load_json(OUT / "mc_results.json")

    if qaoa.get("k_iterations") != mc.get("k_iterations"):
        import warnings
        warnings.warn(
            f"QAOA k={qaoa.get('k_iterations')} != MC k={mc.get('k_iterations')}. "
            "Comparison is only fair when both use the same iteration budget. "
            "Re-run both at k=50 for paper-comparable results."
        )

    qaoa_auc = _auc(qaoa["cumulative_failures"])
    mc_auc = _auc(mc["cumulative_failures"])
    qaoa_ttff = _time_to_first_failure(qaoa["cumulative_failures"])
    mc_ttff = _time_to_first_failure(mc["cumulative_failures"])

    comparison = {
        "question": "Can QAOA-assisted exploration surface unsafe traffic separation behaviors faster than Monte Carlo?",
        "k_iterations": qaoa["k_iterations"],
        "per_iteration": {
            "qaoa": qaoa["cumulative_failures"],
            "monte_carlo": mc["cumulative_failures"],
        },
        "summary": {
            "qaoa_unique_failures": qaoa["total_unique_failures"],
            "mc_unique_failures": mc["total_unique_failures"],
            "qaoa_auc": qaoa_auc,
            "mc_auc": mc_auc,
            "auc_advantage_pct": round(((qaoa_auc - mc_auc) / mc_auc) * 100.0, 2) if mc_auc else 0.0,
            "qaoa_time_to_first_failure": qaoa_ttff,
            "mc_time_to_first_failure": mc_ttff,
            "qaoa_failure_types": _failure_breakdown(qaoa["failures"]),
            "mc_failure_types": _failure_breakdown(mc["failures"]),
            "mean_unique_failures": round(mean([qaoa["total_unique_failures"], mc["total_unique_failures"]]), 3),
            "std_unique_failures": round(pstdev([qaoa["total_unique_failures"], mc["total_unique_failures"]]), 3),
        },
        "winner": "QAOA" if qaoa["total_unique_failures"] >= mc["total_unique_failures"] else "MonteCarlo",
        "qaoa_circuit_info": qaoa["circuit_info"],
        "top_failures": qaoa["failures"][:10],
        "circuit_text": qaoa["circuit_text"],
    }

    (OUT / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison


def main() -> None:
    results = compare_results()
    qtrd = compare_qtrd_vs_mc()
    print(f"Winner             : {results['winner']}")
    print(f"QTRD vs MC winner  : {qtrd['winner']}")
    print(f"Comparison written : {OUT / 'comparison.json'}")


if __name__ == "__main__":
    main()
