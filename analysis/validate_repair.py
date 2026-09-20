"""Validates the repaired files against outputs persisted before the truncation.
Writes outputs/repair_validation.json.  Run from the repository root (about 3 min).
"""
from __future__ import annotations
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import run_mechanism_controls as R  # noqa: E402
from phase3_quantum_tree.classical_layer import run_classical_layer  # noqa: E402
from phase3_quantum_tree.pipeline import _select_evaluator  # noqa: E402
from phase3_quantum_tree.quantum_layer import QuantumDistillationLayer  # noqa: E402
OUT = ROOT / "outputs"

def main() -> None:
    report = {}
    # 1. per-candidate circuit scores, seed 42
    c = run_classical_layer(seed=42, evaluator=_select_evaluator("geometric"))
    q = QuantumDistillationLayer(shots=1024, seed=42, variant="full").refine(c["hidden"], c["teacher_probs"])
    key = lambda p: (p["int_heading"], p["int_altitude"], p["int_speed"], p["int_x_offset"])
    mine = {key(s): float(q["quantum_scores"][i]) for i, s in enumerate(c["states"])}
    pool = json.loads((OUT / "ranking_diagnostic_seed42.json").read_text())["qtd"]["pool_ranking_150"]
    diffs = [abs(mine[key(r["params"])] - r["quantum_score"]) for r in pool]
    report["seed42_per_candidate_scores"] = {"candidates": len(pool), "gated_nonzero": sum(r["quantum_score"] > 0 for r in pool),
                                             "max_abs_difference": max(diffs), "match": max(diffs) < 1e-6}
    # 2. per-seed curves of every arm vs the first export
    orig = json.loads((OUT / "mechanism_controls_original_export.json").read_text())["arms"]
    report["mechanism_arms_curves_identical_seeds"] = {
        arm: sum(R.run_arm(row["seed"], arm)["cumulative_failures"] == row["cumulative_failures"] for row in orig[arm]["per_seed"])
        for arm in R.ARMS}
    # 3. repaired split script vs persisted 1024-shot results
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, "run_split_experiment.py", "--shots", "1024", "--output-dir", tmp,
                        "--output-name", "split_results_1024.json"], cwd=ROOT, check=True, capture_output=True)
        new = json.loads((Path(tmp) / "split_results_1024.json").read_text())
    old = json.loads((OUT / "split_results_1024.json").read_text())
    report["split_experiment"] = {
        "aggregate_identical": new["aggregate"] == old["aggregate"],
        "per_split_curves_identical": sum(a["qtd_cumulative_failures"] == b["qtd_cumulative_failures"] and
                                          a["teacher_only_cumulative_failures"] == b["teacher_only_cumulative_failures"]
                                          for a, b in zip(new["per_split"], old["per_split"])),
        "note": "MC rows in this file are a secondary in-file estimate; mc_no_replacement.json is canonical."}
    (OUT / "repair_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
