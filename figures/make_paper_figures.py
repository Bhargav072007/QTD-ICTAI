#!/usr/bin/env python3
"""Regenerate every figure of the camera-ready paper from canonical outputs/*.json.

Empirical plotted values are loaded from JSON or recomputed from the encounter
model. Circuit/architecture schematics contain explicitly stated design constants.
Fonts are embedded as TrueType (pdf.fonttype 42), so the PDFs contain no Type 3 fonts.

Run from the repository root:  python figures/make_paper_figures.py
Outputs: figures/*.pdf
"""
from __future__ import annotations

import json
import argparse
import hashlib
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs"
FIG = ROOT / "outputs/reproduction/figures"
OVERLAY = ROOT / "outputs/reproduction/qaoa_corrected"
INPUTS = {}

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    # Liberation Serif matches the camera-ready body font and is what the
    # submitted figures were rendered with; DejaVu Serif ships with matplotlib
    # and is the portable fallback.  figure_provenance.json records which was used.
    "font.serif": ["Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "axes.grid": True, "grid.color": "0.88", "grid.linewidth": 0.5,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.spines.top": False, "axes.spines.right": False,
})
COL = 3.45  # IEEE column width (in)
C = {"qtd": "#1f5f99", "teacher": "#d98e04", "cem": "#2a8c6a", "mc": "#555555",
     "qaoa": "#a0447d", "cold": "#7fb2d9", "coldt": "#f0c46a", "exact": "#8c5a2b", "oracle": "#b23a3a"}


def r1(x: float) -> str:
    """Round half away from zero to one decimal (matches the paper's rounding)."""
    from decimal import ROUND_HALF_UP, Decimal
    return str(Decimal(repr(round(x, 6))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def save(fig, name: str) -> None:
    fig.savefig(FIG / name, metadata={"CreationDate": None, "ModDate": None})


def load(name: str):
    path = OVERLAY / name if OVERLAY and (OVERLAY / name).exists() else OUT / name
    INPUTS[name] = {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return json.loads(path.read_text(encoding="utf-8"))


def ms(values):
    a = np.asarray(values, float)
    return float(a.mean()), float(a.std())


def dots(ax, x, values, color, rng, width=0.13):
    jitter = rng.uniform(-width, width, size=len(values))
    ax.scatter(np.full(len(values), x) + jitter, values, s=9, color=color, alpha=0.55, linewidths=0, zorder=2)
    m, s = ms(values)
    ax.errorbar([x + 0.27], [m], yerr=[s], fmt="o", ms=3.2, color="black", ecolor="black", elinewidth=0.9, capsize=2.2, zorder=3)
    return m, s


# ----------------------------------------------------------------- Fig: geometry
def fig_failure_geometry():
    from phase2_qaoa.qaoa_runner import evaluate_state
    from phase2_qaoa.qubo_encoder import PARAM_GRID, enumerate_parameter_states

    H, A, V, D = (PARAM_GRID[k] for k in ("int_heading", "int_altitude", "int_speed", "int_x_offset"))
    hd = np.zeros((4, 4), int)
    av = np.zeros((4, 4), int)
    for s in enumerate_parameter_states():
        if evaluate_state(s)["failure"]:
            hd[H.index(s["int_heading"]), D.index(s["int_x_offset"])] += 1
            av[A.index(s["int_altitude"]), V.index(s["int_speed"])] += 1
    assert hd.sum() == 18 and av.sum() == 18
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.55))
    for ax, mat, xt, yt, xl, yl in (
        (axes[0], hd, [f"{int(v)}" for v in D], [f"{int(v)}" for v in H], "lateral offset (nm)", "heading (deg)"),
        (axes[1], av, [f"{int(v)}" for v in V], [f"{int(v / 1000)}k" for v in A], "speed (nm/step)", "altitude (ft)"),
    ):
        ax.imshow(mat, origin="lower", cmap="Blues", vmin=0, vmax=max(3, mat.max()), aspect="auto")
        ax.grid(False)
        for i in range(4):
            for j in range(4):
                if mat[i, j]:
                    ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=8,
                            color="white" if mat[i, j] >= 2 else "black")
        ax.set_xticks(range(4), xt)
        ax.set_yticks(range(4), yt)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        for sp in ax.spines.values():
            sp.set_visible(True)
    fig.subplots_adjust(wspace=0.55)
    save(fig, "fig_failure_geometry.pdf")
    plt.close(fig)


# ------------------------------------------------------------- Fig: architecture
def fig_architecture():
    fig, ax = plt.subplots(figsize=(COL, 1.25))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 36)
    ax.axis("off")
    boxes = [(6, "Layer 1\nTeacher", "MLP 4-8-1\nall 256 labels", "0.93"),
             (39, "Layer 2\nRefiner", "3-qubit $R_Y$/$CZ$\ntop-25% by $p_T$", "0.85"),
             (72, "Layer 3\nStudent", "logistic reg.\nranked eval.", "0.93")]
    w, h, y0 = 24, 22, 9
    for x, title, sub, shade in boxes:
        ax.add_patch(mpatches.FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                             linewidth=0.9, edgecolor="black", facecolor=shade))
        ax.text(x + w / 2, y0 + h - 5.6, title, ha="center", va="center", fontsize=7.8, fontweight="bold")
        ax.text(x + w / 2, y0 + 5.6, sub, ha="center", va="center", fontsize=7)

    def arrow(x1, x2, label):
        ax.annotate("", xy=(x2, y0 + h / 2), xytext=(x1, y0 + h / 2),
                    arrowprops=dict(arrowstyle="-|>", lw=1.0, color="black"))
        ax.text((x1 + x2) / 2, y0 + h / 2 + 2.6, label, ha="center", fontsize=7.5)

    arrow(0, 6, "")
    arrow(30, 39, "$p_T, h$")
    arrow(63, 72, "$q_s$")
    arrow(96, 100, "")
    ax.text(51, 3.0, r"$\tilde{y}=\lambda(p_T)\,p_T+(1-\lambda(p_T))\,q_s$", ha="center", fontsize=8)
    save(fig, "fig_architecture.pdf")
    plt.close(fig)


def fig_circuit():
    fig, b = plt.subplots(figsize=(COL * 0.92, 1.35))
    b.set_xlim(0, 12)
    b.set_ylim(0.6, 9.4)
    b.axis("off")
    ys = [7.8, 5.0, 2.2]
    for i, yy in enumerate(ys):
        b.plot([0.7, 11.3], [yy, yy], color="black", lw=0.9)
        b.text(0.15, yy, f"$q_{i}$", ha="center", va="center", fontsize=9)

    def gate(x, yy, txt, w=1.8, fc="0.9"):
        b.add_patch(mpatches.Rectangle((x - w / 2, yy - 0.9), w, 1.8, facecolor=fc, edgecolor="black", lw=0.9, zorder=3))
        b.text(x, yy, txt, ha="center", va="center", fontsize=8, zorder=4)

    for i, yy in enumerate(ys):
        gate(2.1, yy, rf"$R_Y(\theta_{i})$")
    for x, (a, c) in ((4.4, (0, 1)), (5.7, (1, 2))):
        b.plot([x, x], [ys[c], ys[a]], color="black", lw=0.9)
        b.scatter([x, x], [ys[a], ys[c]], color="black", s=18, zorder=3)
    gate(7.8, ys[0], r"$R_Y(\pi p_T)$", w=2.3)
    for yy in ys:
        gate(10.3, yy, "meas.", w=1.5, fc="white")
    b.text(11.5, 5.0, "$q_s$", ha="left", va="center", fontsize=9)
    save(fig, "fig_circuit.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ Fig: QAOA
def fig_qaoa():
    rng = np.random.default_rng(0)
    qaoa = [r["unique_failures"] for r in load("qaoa_multiseed.json")["per_seed"]]
    mc = [r["unique_failures"] for r in load("mc_no_replacement.json")["table_ii_mc_records"]]
    # Paper Fig. 4: the exact energy ranking is read down to EACH SEED's own
    # unique-state budget, so it is budget-matched to QAOA rather than fixed at 50.
    matched = load("qubo_diagnostics.json")["qaoa_matched_unique_budgets"]
    exact_matched = [r["energy_ranking_failures"] for r in matched["per_seed"]]
    fig, ax = plt.subplots(figsize=(COL, 1.75))
    for x, vals, col in ((0, qaoa, C["qaoa"]), (1, mc, C["mc"]), (2, exact_matched, C["exact"])):
        m, s = dots(ax, x, vals, col, rng)
        ax.text(x + 0.36, m, r1(m) + "$\\pm$" + r1(s), va="center", fontsize=7.5)
    ax.set_xticks([0, 1, 2], ["QAOA ($p{=}2$)\n400 calls/seed", "Monte Carlo\n50 states",
                              "Exact QUBO\nrank (matched)"])
    ax.set_xlim(-0.5, 3.25)
    ax.set_ylim(0, 9)
    ax.set_ylabel(f"Failures found (of {load('policy_failure_overlap.json')['geometric_failures']})")
    save(fig, "fig_qaoa_multiseed.pdf")
    plt.close(fig)


# -------------------------------------------------------- Fig: mechanism controls
def fig_mechanism():
    rng = np.random.default_rng(1)
    mech = load("mechanism_controls.json")["paired_comparisons"]
    pos = load("positive_control.json")["readouts"]["student"]["paired_comparisons"]
    arms = [("qtd_full", "Full\ncircuit", mech), ("no_cz", "No $CZ$", mech), ("analytic", "Exact\n(no shots)", mech),
            ("matched_classical", "Matched\nclass.", mech), ("shuffled", "Shuffled", mech),
            ("random", "Random", mech), ("constant", "Const.\n0.5", pos), ("zero_blend", "Zero\nscore", pos)]
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    for x, (key, _, src) in enumerate(arms):
        vals = src[key]["vs_teacher_only"]["auc_delta"]["per_seed"]
        col = C["qtd"] if src is mech else C["oracle"]
        m, _ = dots(ax, x, vals, col, rng, width=0.10)
        ax.text(x + 0.05, 71, ("+" if m > 0 else "") + r1(m), ha="center", va="center", fontsize=6.3)
    ax.axhline(0, color="0.4", lw=0.7, ls="--")
    # Labels carry their own line breaks, so keep them horizontal: rotating them
    # costs vertical space and reads worse for two-line entries.
    ax.set_xticks(range(len(arms)), [a[1] for a in arms], fontsize=6.6)
    ax.set_xlim(-0.5, len(arms) - 0.25)
    ax.set_ylim(-55, 78)
    ax.set_ylabel("AUC difference vs.\nteacher-only")
    save(fig, "fig_mechanism_controls.pdf")
    plt.close(fig)


# ---------------------------------------------------------- Fig: policy-conditioned
def fig_policy():
    pol = load("policy_loop_comparison.json")["per_seed"]
    mc = [r["unique_failures"] for r in load("policy_loop_norepl.json")["mc_records"]]
    cem = [r["unique_failures"] for r in load("adaptive_baseline_policy.json")["records"]]
    series = [("QTD\n(full circuit)", [r["qtd_quantum_on"]["unique_failures"] for r in pol], C["qtd"]),
              ("Teacher-only", [r["teacher_only_no_quantum"]["unique_failures"] for r in pol], C["teacher"]),
              ("CEM\n(label-free)", cem, C["cem"]), ("Monte Carlo", mc, C["mc"])]
    fig, ax = plt.subplots(figsize=(COL, 1.7))
    for x, (_, vals, col) in enumerate(series):
        m, s = ms(vals)
        ax.bar(x, m, yerr=s, color=col, width=0.62, error_kw=dict(lw=0.9, capsize=2.5), zorder=2)
        ax.text(x, m + s + 0.45, r1(m), ha="center", fontsize=7.5)
    ax.axhline(load("policy_failure_overlap.json")["policy_failures"], color="0.35", lw=0.7, ls=":")
    ax.text(3.45, 12.25, f"{load('policy_failure_overlap.json')['policy_failures']} policy-failing states", ha="right", fontsize=7)
    ax.set_xticks(range(4), [s[0] for s in series])
    ax.set_ylim(0, 15)
    ax.set_yticks([0, 3, 6, 9, 12])
    ax.set_ylabel(f"Failing states found\n(of {load('policy_failure_overlap.json')['policy_failures']})")
    save(fig, "fig_policy_conditioned.pdf")
    plt.close(fig)


# ------------------------------------------------------------- Fig: cumulative (seed 42)
def fig_cumulative():
    d = load("fig_cumulative_norepl.json")
    k = np.arange(1, 51)
    fig, ax = plt.subplots(figsize=(COL, 1.7))
    ax.step(k, d["qtd_cum42"], where="post", color=C["qtd"], lw=1.4, label="QTD (full circuit)")
    ax.step(k, d["mc_cum42"], where="post", color=C["mc"], lw=1.3, ls="--", label="Monte Carlo")
    ax.axhline(load("policy_failure_overlap.json")["geometric_failures"], color="0.35", lw=0.7, ls=":")
    ax.text(1, 18.4, f"all {load('policy_failure_overlap.json')['geometric_failures']} failing states", fontsize=7)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 20.5)
    ax.set_xlabel("Evaluations used")
    ax.set_ylabel("Cumulative failing states")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62))
    save(fig, "fig_cumulative.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Fig: budget curves
def fig_budget_curves():
    splits = load("split_results_1024.json")["per_split"]
    mc = load("mc_no_replacement.json")["experiment_5_mc_records"]
    k = np.arange(1, 51)
    fig, ax = plt.subplots(figsize=(COL, 1.8))
    for label, curves, col, ls in (
        ("QTD (full circuit)", [np.array(r["qtd_cumulative_failures"]) / r["test_failure_count"] for r in splits], C["qtd"], "-"),
        ("Teacher-only", [np.array(r["teacher_only_cumulative_failures"]) / r["test_failure_count"] for r in splits], C["teacher"], "--"),
        ("Monte Carlo", [np.array(r["cumulative_failures"]) / r["test_failures"] for r in mc], C["mc"], ":"),
    ):
        arr = np.vstack(curves)
        m, s = arr.mean(0), arr.std(0)
        ax.plot(k, m, color=col, ls=ls, lw=1.3, label=label)
        ax.fill_between(k, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), color=col, alpha=0.15, lw=0)
    ax.set_xlim(1, 50)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("Evaluation budget $k$")
    ax.set_ylabel("Held-out recall (of 9)")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.76))
    save(fig, "fig_budget_curves.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ Fig: lift
def fig_lift():
    rng = np.random.default_rng(2)
    ins = load("mechanism_controls.json")["paired_comparisons"]["qtd_full"]["vs_teacher_only"]["auc_delta"]["per_seed"]
    held = [r["qtd_auc"] - r["teacher_only_auc"] for r in load("split_results_1024.json")["per_split"]]
    fig, ax = plt.subplots(figsize=(COL, 1.7))
    for x, vals, col in ((0, ins, C["qtd"]), (1, held, C["teacher"])):
        m, s = dots(ax, x, vals, col, rng)
        ax.text(x + 0.36, m, "+" + r1(m) + "$\\pm$" + r1(s), va="center", fontsize=7.5)
    ax.axhline(0, color="0.4", lw=0.7, ls="--")
    ax.set_xticks([0, 1], ["In-sample\n(10 seeds)", "Label-disjoint\n(10 splits)"])
    ax.set_xlim(-0.5, 1.9)
    ax.set_ylabel("Full circuit minus\nteacher-only AUC")
    save(fig, "fig_quantum_lift.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Fig: regimes
def fig_regimes():
    abl = load("quantum_ablation_multiseed.json")["per_seed"]
    cold = load("coldstart_qtd.json")["per_seed"]
    exact = load("qubo_diagnostics.json")["variants"]["with_penalty"]["upper_triangular_hamiltonian"]["failures_in_lowest_energy"]["50"]
    rows = [
        ("QTD (full circuit)", [r["qtd_quantum_on"]["unique_failures"] for r in abl], C["qtd"]),
        ("Teacher-only", [r["teacher_only_no_quantum"]["unique_failures"] for r in abl], C["teacher"]),
        None,
        ("CEM (label-free)", [r["unique_failures"] for r in load("adaptive_baseline.json")["records"]], C["cem"]),
        ("Monte Carlo", [r["unique_failures"] for r in load("mc_no_replacement.json")["table_ii_mc_records"]], C["mc"]),
        None,
        ("Exact QUBO rank, 50 states$^{\\ddagger}$", [exact], C["exact"]),
        ("QAOA direct$^{\\dagger}$", [r["unique_failures"] for r in load("qaoa_multiseed.json")["per_seed"]], C["qaoa"]),
        None,
        ("Cold-start teacher-only", [r["coldstart_teacher_only"]["unique_failures"] for r in cold], C["coldt"]),
        ("Cold-start QTD", [r["coldstart_qtd"]["unique_failures"] for r in cold], C["cold"]),
    ]
    fig, ax = plt.subplots(figsize=(COL, 2.35))
    y, ticks, labels = 0.0, [], []
    for row in rows:
        if row is None:
            y -= 0.45
            continue
        name, vals, col = row
        m, s = ms(vals)
        ax.barh(y, m, xerr=s if len(vals) > 1 else None, color=col, height=0.72,
                error_kw=dict(lw=0.9, capsize=2.2), zorder=2)
        txt = r1(m) + "$\\pm$" + r1(s) if len(vals) > 1 else f"{m:.0f}"
        ax.text(m + (s if len(vals) > 1 else 0) + 0.4, y, txt, va="center", fontsize=7.2)
        ticks.append(y)
        labels.append(name)
        y -= 1.0
    ax.axvline(load("policy_failure_overlap.json")["geometric_failures"], color="0.35", lw=0.7, ls=":")
    ax.set_yticks(ticks, labels)
    ax.set_xlim(0, 23.5)
    ax.set_xticks([0, 5, 10, 15, 18])
    ax.set_xlabel(f"Failing states found (of {load('policy_failure_overlap.json')['geometric_failures']})")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_budget_regimes.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIG)
    parser.add_argument("--data-dir", type=Path, default=OVERLAY, help="Overlay corrected JSON (default: tracked corrected QAOA artifacts); unchanged sources use historical outputs")
    args = parser.parse_args()
    FIG, OVERLAY = args.output_dir, args.data_dir
    FIG.mkdir(parents=True, exist_ok=False)
    for fn in (fig_failure_geometry, fig_architecture, fig_circuit, fig_qaoa, fig_mechanism, fig_policy,
               fig_cumulative, fig_budget_curves, fig_lift, fig_regimes):
        fn()
        print("ok", fn.__name__)

    from matplotlib import font_manager
    # Record the font actually resolved for the serif family, not a fixed name.
    font_path = Path(font_manager.findfont(font_manager.FontProperties(family="serif")))
    (FIG / "figure_provenance.json").write_text(json.dumps({
        "python": platform.python_version(), "matplotlib": matplotlib.__version__,
        "font": font_path.name, "font_sha256": hashlib.sha256(font_path.read_bytes()).hexdigest(),
        "inputs": INPUTS, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2))
