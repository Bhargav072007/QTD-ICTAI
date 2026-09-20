"""Checks every number that is new or changed in the camera-ready text against outputs/*.json.
(run_traceability_checks.py covers the numbers carried over from the reviewed version.)
Exit code 0 only if every check passes.  Run from the repository root.
"""
from __future__ import annotations
import json, sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
OUT = Path(__file__).resolve().parent.parent / "outputs"
L = lambda n: json.loads((OUT / n).read_text())

def rnd(x, d):
    return float(Decimal(repr(round(float(x), 8))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))

fails = 0
def chk(name, paper, actual, d=1):
    global fails
    ok = abs(rnd(actual, d) - paper) < 1e-9
    fails += (not ok)
    print(f"{'PASS' if ok else 'FAIL'} | {name} | paper {paper} | actual {actual}")

mech = L("mechanism_controls.json")["paired_comparisons"]
for arm, m, s in (("qtd_full", 26.5, 12.2), ("no_cz", 28.0, 15.0), ("analytic", 26.5, 12.3),
                  ("matched_classical", 28.9, 15.1), ("shuffled", 26.7, 14.4), ("random", 14.1, 3.6)):
    a = mech[arm]["vs_teacher_only"]["auc_delta"]["aggregate"]
    chk(f"mechanism {arm} AUC diff mean", m, a["mean"]); chk(f"mechanism {arm} AUC diff sd", s, a["std"])
    chk(f"mechanism {arm} positive seeds", 10, mech[arm]["vs_teacher_only"]["auc_delta"]["exact_tests"]["positive_count"], 0)
f = mech["qtd_full"]["vs_teacher_only"]
chk("full AUC CI low", 19.6, f["auc_delta"]["bootstrap_mean_95_ci"]["ci_95_low"]); chk("full AUC CI high", 34.7, f["auc_delta"]["bootstrap_mean_95_ci"]["ci_95_high"])
chk("count diff mean", 1.3, f["unique_failure_delta"]["aggregate"]["mean"]); chk("count diff sd", 1.2, f["unique_failure_delta"]["aggregate"]["std"])
chk("count CI low", 0.6, f["unique_failure_delta"]["bootstrap_mean_95_ci"]["ci_95_low"]); chk("count CI high", 2.0, f["unique_failure_delta"]["bootstrap_mean_95_ci"]["ci_95_high"])
chk("count sign p", 0.03, f["unique_failure_delta"]["exact_tests"]["sign_test_p"], 2)
chk("count positive", 6, f["unique_failure_delta"]["exact_tests"]["positive_count"], 0); chk("count ties", 4, f["unique_failure_delta"]["exact_tests"]["zero_count"], 0)
for arm, v in (("no_cz", 1.3), ("analytic", 1.3), ("matched_classical", 1.4), ("shuffled", 1.4)):
    chk(f"{arm} count diff", v, mech[arm]["vs_teacher_only"]["unique_failure_delta"]["aggregate"]["mean"])

pc = L("positive_control.json")["readouts"]
st, tg = pc["student"]["paired_comparisons"], pc["target"]["paired_comparisons"]
a = st["constant"]["vs_teacher_only"]["auc_delta"]; chk("constant mean", 20.0, a["aggregate"]["mean"]); chk("constant sd", 7.8, a["aggregate"]["std"])
a = st["zero_blend"]["vs_teacher_only"]["auc_delta"]; chk("zero mean", -27.9, a["aggregate"]["mean"]); chk("zero sd", 14.1, a["aggregate"]["std"]); chk("zero positive seeds", 0, a["exact_tests"]["positive_count"], 0)
a = st["qtd_full"]["vs_constant"]["auc_delta"]; chk("full-constant mean", 6.5, a["aggregate"]["mean"]); chk("full-constant sd", 6.3, a["aggregate"]["std"])
chk("full-constant CI low", 2.6, a["bootstrap_mean_95_ci"]["ci_95_low"]); chk("full-constant CI high", 10.6, a["bootstrap_mean_95_ci"]["ci_95_high"]); chk("full-constant positive", 9, a["exact_tests"]["positive_count"], 0)
table = {"qtd_full": ((0.1, 6.3, 4), (-10.1, 58.7, 3)), "oracle_025": ((2.3, 8.1, 7), (181.6, 59.2, 10)),
         "oracle_050": ((3.7, 6.0, 6), (304.5, 46.9, 10)), "oracle_100": ((6.9, 7.5, 8), (336.8, 55.0, 10))}
for arm, (s_, t_) in table.items():
    for tag, src, (m, sd, pos) in (("student", st, s_), ("target", tg, t_)):
        a = src[arm]["vs_own_shuffle"]["auc_delta"]
        chk(f"Table IV {arm} {tag} mean", m, a["aggregate"]["mean"]); chk(f"Table IV {arm} {tag} sd", sd, a["aggregate"]["std"]); chk(f"Table IV {arm} {tag} positive", pos, a["exact_tests"]["positive_count"], 0)
a = st["oracle_100"]["vs_own_shuffle"]["auc_delta"]["bootstrap_mean_95_ci"]; chk("oracle CI low", 2.8, a["ci_95_low"]); chk("oracle CI high", 11.8, a["ci_95_high"])
a = tg["qtd_full"]["vs_teacher_only"]["auc_delta"]["aggregate"]; chk("target readout full-teacher mean", -35.0, a["mean"]); chk("target readout full-teacher sd", 51.5, a["std"])

q = L("qubo_diagnostics.json")["variants"]; w = q["with_penalty"]["xTQx_symmetric"]
chk("QUBO AUC", 0.75, w["ranking_auc"], 2); chk("QUBO AUC (Hamiltonian)", 0.75, q["with_penalty"]["upper_triangular_hamiltonian"]["ranking_auc"], 2)
chk("QUBO AUC no penalty", 0.74, q["without_penalty"]["xTQx_symmetric"]["ranking_auc"], 2)
for k, v in (("heading", 0.46), ("altitude", 0.63), ("speed", 0.68), ("lateral_offset", 0.54)):
    chk(f"marginal AUC {k}", v, w["per_parameter"][k]["marginal_ranking_auc"], 2)
for n, v in (("18", 2), ("45", 7), ("50", 7)):
    chk(f"failures in lowest {n}", v, w["failures_in_lowest_energy"][n], 0)
    chk(f"failures in lowest {n} (Hamiltonian)", v, q["with_penalty"]["upper_triangular_hamiltonian"]["failures_in_lowest_energy"][n], 0)
chk("global min is failure", 0, int(w["global_minimum_is_failure"]), 0); chk("exact first failure", 8, w["rank_of_first_failure"], 0)
cum = w["exact_enumeration_cumulative_failures_k50"]; prev = t = 0
for x in cum: t += (prev + x) / 2; prev = x
chk("exact minimizer AUC", 150.5, t); chk("heading index-0 failures", 12, w["per_parameter"]["heading"]["failures_by_value_index"][0], 0)
chk("speed failures 4/5/6", 396, int("".join(str(v) for v in w["per_parameter"]["speed"]["failures_by_value_index"][1:])), 0)

o = L("policy_failure_overlap.json")
for k, v in (("policy_failures", 12), ("policy_failures_also_geometric", 3), ("policy_failures_safe_under_fixed_ego", 9),
             ("warmstart_records", 26), ("warmstart_unique_states", 12)):
    chk(k, v, o[k], 0)
chk("warm start equals policy failures", 1, int(o["warmstart_equals_policy_failure_set"]), 0)
chk("policy action constant", 256, o["policy_primary_action_counts_over_256_states"].get("turn_right", 0), 0)

s = L("camera_ready_stats.json")
k20 = s["heldout_recall_difference_qtd_minus_teacher"]["k20"]
chk("k20 diff mean", 0.06, k20["mean"], 2); chk("k20 diff sd", 0.12, k20["pstd"], 2); chk("k20 positive", 5, k20["positive"], 0); chk("k20 negative", 2, k20["negative"], 0); chk("k20 p", 0.45, k20["sign_test_p"], 2)
t_ = s["teacher_in_sample"]; chk("teacher AUROC", 0.925, t_["ranking_auroc_mean"], 3); chk("teacher AUROC sd", 0.001, t_["ranking_auroc_pstd"], 3)
chk("teacher min acc", 0.918, min(x["accuracy"] for x in t_["per_seed"]), 3); chk("teacher max acc", 0.930, max(x["accuracy"] for x in t_["per_seed"]), 3)
chk("all failures inside gate (min over seeds)", 18, min(x["failures_in_top25pct_gate"] for x in t_["per_seed"]), 0)

ab = L("quantum_ablation_multiseed.json")["aggregate"]["teacher_only_no_quantum"]
chk("teacher-only AUC", 351.7, ab["auc"]["mean"]); chk("teacher-only AUC sd", 28.9, ab["auc"]["std"]); chk("teacher-only first", 15.2, ab["time_to_first_failure"]["mean"]); chk("teacher-only first sd", 1.7, ab["time_to_first_failure"]["std"])
v3 = L("reproduction/qtd_v3_stochastic_20260713T060331Z/qtd_v3_stochastic_suite.json")["decision"]
for paper, act in zip((-0.033, -0.028, 0.014, -0.033), v3["per_regime_delta"]): chk("QTDv3 regime delta", paper, act, 3)
chk("QTDv3 mean", -0.020, v3["aggregate"]["mean"], 3); chk("QTDv3 CI low", -0.033, v3["bootstrap_mean_95_ci"]["ci_95_low"], 3); chk("QTDv3 CI high", 0.002, v3["bootstrap_mean_95_ci"]["ci_95_high"], 3)
v2 = L("reproduction/qtd_v2_matched_20260713T055424Z/qtd_v2_matched_benchmark.json")["decision"]
chk("QTDv2 mean", -75.4, v2["primary_auc_delta"]["aggregate"]["mean"]); chk("QTDv2 CI low", -107.1, v2["primary_auc_delta"]["bootstrap_mean_95_ci"]["ci_95_low"]); chk("QTDv2 CI high", -40.0, v2["primary_auc_delta"]["bootstrap_mean_95_ci"]["ci_95_high"]); chk("QTDv2 positive", 1, v2["positive_auc_delta_count"], 0)
print(f"\n{'ALL PASS' if fails == 0 else str(fails) + ' FAIL'}")
sys.exit(1 if fails else 0)
