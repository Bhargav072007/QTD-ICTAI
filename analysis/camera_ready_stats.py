"""Small derived statistics cited in the camera-ready text.  Classical only.
Output: outputs/camera_ready_stats.json
"""
from __future__ import annotations
import json, statistics, sys
from math import comb
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from phase3_quantum_tree.classical_layer import run_classical_layer  # noqa: E402
from phase3_quantum_tree.pipeline import _select_evaluator  # noqa: E402


def sign_p(values):
    pos = sum(v > 0 for v in values); neg = sum(v < 0 for v in values); n = pos + neg
    if n == 0:
        return pos, neg, None
    return pos, neg, round(min(1.0, 2 * sum(comb(n, i) for i in range(min(pos, neg) + 1)) / 2**n), 6)


def rank_auc(score, labels):
    pos, neg = score[labels == 1], score[labels == 0]
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def main():
    out = {}
    splits = json.loads((ROOT / "outputs" / "split_results_1024.json").read_text())["per_split"]
    out["heldout_recall_difference_qtd_minus_teacher"] = {}
    for k in (10, 20, 30):
        d = [(r["qtd_cumulative_failures"][k - 1] - r["teacher_only_cumulative_failures"][k - 1]) / r["test_failure_count"] for r in splits]
        pos, neg, p = sign_p(d)
        out["heldout_recall_difference_qtd_minus_teacher"][f"k{k}"] = {
            "per_split": [round(x, 6) for x in d], "mean": round(statistics.mean(d), 6),
            "pstd": round(statistics.pstdev(d), 6), "positive": pos, "negative": neg, "sign_test_p": p}
    teacher = []
    for seed in range(42, 52):
        c = run_classical_layer(seed=seed, evaluator=_select_evaluator("geometric"))
        tp, lab = np.asarray(c["teacher_probs"], float), np.asarray(c["labels"], int)
        teacher.append({"seed": seed, "accuracy": round(float(((tp > 0.5).astype(int) == lab).mean()), 6),
                        "ranking_auroc": round(rank_auc(tp, lab), 6),
                        "max_p_on_failures": round(float(tp[lab == 1].max()), 6),
                        "failures_in_top25pct_gate": int(lab[np.argsort(tp)[::-1][:64]].sum())})
    out["teacher_in_sample"] = {"majority_class_accuracy": round(238 / 256, 6), "per_seed": teacher,
                                "ranking_auroc_mean": round(statistics.mean(t["ranking_auroc"] for t in teacher), 6),
                                "ranking_auroc_pstd": round(statistics.pstdev(t["ranking_auroc"] for t in teacher), 6)}
    (ROOT / "outputs" / "camera_ready_stats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out["heldout_recall_difference_qtd_minus_teacher"].items()}, indent=0))
    print(out["teacher_in_sample"]["ranking_auroc_mean"], out["teacher_in_sample"]["ranking_auroc_pstd"], teacher[0])


if __name__ == "__main__":
    main()
