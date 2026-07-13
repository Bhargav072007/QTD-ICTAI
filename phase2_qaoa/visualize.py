"""
QAA Phase 2 - self-contained HTML dashboard generator.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase2_qaoa.qaoa_runner import OUT


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _line_chart_svg(
    qaoa: List[int],
    mc: List[int],
    qtrd: List[int] | None = None,
    width: int = 560,
    height: int = 240,
) -> str:
    left = 48
    right = width - 20
    top = 20
    bottom = height - 36
    qtrd_values = qtrd or []
    max_value = max(max(qaoa, default=0), max(mc, default=0), max(qtrd_values, default=0), 1)

    def point(index: int, value: int, total: int) -> str:
        x = left + (right - left) * (index / max(total - 1, 1))
        y = bottom - (bottom - top) * (value / max_value)
        return f"{x:.1f},{y:.1f}"

    qaoa_points = " ".join(point(i, value, len(qaoa)) for i, value in enumerate(qaoa))
    mc_points = " ".join(point(i, value, len(mc)) for i, value in enumerate(mc))
    qtrd_polyline = ""
    qtrd_legend = ""
    if qtrd_values:
        qtrd_points = " ".join(point(i, value, len(qtrd_values)) for i, value in enumerate(qtrd_values))
        qtrd_polyline = f'<polyline points="{qtrd_points}" fill="none" stroke="#0f766e" stroke-width="3"/>'
        qtrd_legend = '<text x="154" y="{legend_y}" fill="#0f766e" font-size="11" font-weight="700">QTRD</text>'
    return f"""
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Cumulative failures chart">
  <rect width="{width}" height="{height}" fill="transparent"/>
  <line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#cbd5e1"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#cbd5e1"/>
  <polyline points="{mc_points}" fill="none" stroke="#94a3b8" stroke-width="3"/>
  <polyline points="{qaoa_points}" fill="none" stroke="#2563eb" stroke-width="3"/>
  {qtrd_polyline}
  <text x="{left + 8}" y="{top + 14}" fill="#2563eb" font-size="11" font-weight="700">QAOA</text>
  <text x="{left + 62}" y="{top + 14}" fill="#64748b" font-size="11" font-weight="700">Monte Carlo</text>
  {qtrd_legend.format(legend_y=top + 14) if qtrd_legend else ''}
  <text x="{width / 2:.0f}" y="{height - 8}" fill="#64748b" font-size="10" text-anchor="middle">Iterations</text>
  <text x="16" y="{height / 2:.0f}" fill="#64748b" font-size="10" text-anchor="middle" transform="rotate(-90 16 {height / 2:.0f})">Unique failures</text>
</svg>
"""


def _bar_chart_svg(summary: Dict[str, Any], width: int = 360, height: int = 220) -> str:
    values = [
        ("QAOA SL", summary["qaoa_failure_types"]["separation_loss"], "#1d4ed8"),
        ("QAOA NM", summary["qaoa_failure_types"]["near_miss"], "#60a5fa"),
        ("MC SL", summary["mc_failure_types"]["separation_loss"], "#475569"),
        ("MC NM", summary["mc_failure_types"]["near_miss"], "#94a3b8"),
    ]
    max_value = max(value for _, value, _ in values) or 1
    bars = []
    for index, (label, value, color) in enumerate(values):
        x = 30 + index * 78
        bar_height = (value / max_value) * 120
        y = 160 - bar_height
        bars.append(
            f'<rect x="{x}" y="{y:.1f}" width="42" height="{bar_height:.1f}" fill="{color}" rx="6"/>'
            f'<text x="{x + 21}" y="{y - 6:.1f}" text-anchor="middle" font-size="10" fill="#334155">{value}</text>'
            f'<text x="{x + 21}" y="186" text-anchor="middle" font-size="10" fill="#64748b">{html.escape(label)}</text>'
        )
    return f"""
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Failure types bar chart">
  <line x1="18" y1="160" x2="{width - 16}" y2="160" stroke="#cbd5e1"/>
  {''.join(bars)}
</svg>
"""


def _table_rows(rows: Iterable[Dict[str, Any]]) -> str:
    rendered = []
    for index, item in enumerate(rows, start=1):
        rendered.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{item['params']['int_heading']:.0f} deg</td>"
            f"<td>{item['params']['int_altitude']:.0f} ft</td>"
            f"<td>{item['params']['int_speed']:.1f}</td>"
            f"<td>{item['params']['int_x_offset']:.0f} NM</td>"
            f"<td>{item['min_h_sep_nm']:.2f} NM</td>"
            f"<td>{item['min_v_sep_ft']:.0f} ft</td>"
            f"<td>{html.escape(item['failure_type'])}</td>"
            "</tr>"
        )
    return "".join(rendered)


def _distilled_rows(rows: Iterable[Dict[str, Any]]) -> str:
    rendered = []
    for index, item in enumerate(rows, start=1):
        rendered.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{item['int_heading']:.0f} deg</td>"
            f"<td>{item['int_altitude']:.0f} ft</td>"
            f"<td>{item['int_speed']:.1f}</td>"
            f"<td>{item['int_x_offset']:.0f} NM</td>"
            f"<td>{item['teacher_prob']:.3f}</td>"
            f"<td>{item['quantum_score']:.3f}</td>"
            f"<td>{item['distilled_target']:.3f}</td>"
            f"<td>{item['hard_label']:.0f}</td>"
            "</tr>"
        )
    return "".join(rendered)


def render_dashboard() -> Path:
    comparison = _load_json(OUT / "comparison.json")
    qtrd_vs_mc_path = OUT / "qtrd_vs_mc_comparison.json"
    qtrd_vs_mc = _load_json(qtrd_vs_mc_path) if qtrd_vs_mc_path.exists() else None
    quantum_tree_path = OUT / "quantum_tree_results.json"
    quantum_tree = _load_json(quantum_tree_path) if quantum_tree_path.exists() else None
    summary = comparison["summary"]
    verdict = "QAOA outperforms Monte Carlo" if comparison["winner"] == "QAOA" else "Monte Carlo remains stronger"
    quantum_tag = ""
    quantum_section = ""
    if quantum_tree is not None:
        quantum_tag = f' <span class="tag alt-tag">Quantum Tree active: {html.escape(quantum_tree["summary"]["quantum_backend"])}</span>'
        architecture_items = "".join(
            f"<li>{html.escape(item)}</li>" for item in quantum_tree["architecture"]
        )
        quantum_section = f"""
    <section class="subgrid four-up">
      <div class="card stats">
        <h2>Quantum Tree - Architecture</h2>
        <ol class="stack-list">{architecture_items}</ol>
      </div>
      <div class="card stats">
        <h2>Quantum Tree - Teacher</h2>
        <p><span>Teacher accuracy</span><strong>{quantum_tree['summary']['teacher_accuracy']}</strong></p>
        <p><span>Teacher loss</span><strong>{quantum_tree['summary']['teacher_loss']}</strong></p>
        <p><span>Mean teacher prob</span><strong>{quantum_tree['summary']['mean_teacher_prob']}</strong></p>
      </div>
      <div class="card stats">
        <h2>Quantum Tree - Quantum Layer</h2>
        <p><span>Backend</span><strong>{html.escape(quantum_tree['summary']['quantum_backend'])}</strong></p>
        <p><span>Mean quantum score</span><strong>{quantum_tree['summary']['mean_quantum_score']}</strong></p>
        <p><span>Student mean score</span><strong>{quantum_tree['summary']['mean_student_score']}</strong></p>
      </div>
      <div class="card stats">
        <h2>Quantum Tree - Student</h2>
        <p><span>Student MSE</span><strong>{quantum_tree['summary']['student_mse']}</strong></p>
        <p><span>Top distilled states</span><strong>{len(quantum_tree['top_distilled_states'])}</strong></p>
        <p><span>Use case</span><strong>Teacher -> Quantum -> Student</strong></p>
      </div>
    </section>

    <section class="card full">
      <h2>Quantum Tree - Top distilled training states</h2>
      <div class="small">These rows are the distilled teacher-plus-quantum signals intended to train the autonomous student model.</div>
      <table>
        <thead>
          <tr>
            <th>#</th><th>Heading</th><th>Altitude</th><th>Speed</th><th>X offset</th><th>Teacher</th><th>Quantum</th><th>Distilled</th><th>Label</th>
          </tr>
        </thead>
        <tbody>{_distilled_rows(quantum_tree['top_distilled_states'])}</tbody>
      </table>
    </section>
"""
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>QAA Quantum Exploration Dashboard</title>
  <style>
    :root {{
      --bg: #eef4ff;
      --ink: #10233f;
      --card: #ffffff;
      --border: #dbe6f4;
      --quantum-tree: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "Segoe UI", Tahoma, sans-serif; background: radial-gradient(circle at top, #f9fbff 0%, var(--bg) 72%); color: var(--ink); }}
    .shell {{ max-width: 1220px; margin: 0 auto; padding: 32px 24px 48px; }}
    .hero {{ background: linear-gradient(135deg, #0f172a, #1d4ed8); color: white; border-radius: 24px; padding: 28px; box-shadow: 0 24px 60px rgba(37, 99, 235, 0.18); }}
    .hero h1 {{ margin: 0 0 8px; font-size: 2rem; }}
    .hero p {{ margin: 0; max-width: 840px; color: rgba(255,255,255,0.82); }}
    .hero .tag {{ display: inline-block; margin-top: 14px; margin-right: 10px; padding: 8px 14px; border-radius: 999px; background: rgba(255,255,255,0.12); font-size: 0.88rem; }}
    .hero .alt-tag {{ background: rgba(16, 185, 129, 0.18); }}
    .verdict {{ margin-top: 20px; display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 14px; }}
    .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 20px; padding: 20px; box-shadow: 0 18px 42px rgba(15, 23, 42, 0.06); }}
    .card h2, .card h3 {{ margin: 0 0 14px; }}
    .grid {{ display: grid; grid-template-columns: 1.35fr 1fr; gap: 18px; margin-top: 18px; }}
    .subgrid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; margin-top: 18px; }}
    .four-up {{ grid-template-columns: repeat(4, 1fr); }}
    .full {{ margin-top: 18px; }}
    .stats p {{ display: flex; justify-content: space-between; gap: 12px; margin: 8px 0; padding-bottom: 8px; border-bottom: 1px solid #edf2f7; }}
    .stats p:last-child {{ border-bottom: none; }}
    .stack-list {{ margin: 0; padding-left: 18px; color: #234; line-height: 1.6; }}
    pre {{ margin: 0; padding: 14px; border-radius: 16px; overflow-x: auto; background: #0f172a; color: #dbeafe; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e5edf7; text-align: left; }}
    th {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; color: #5b6b82; }}
    .small {{ color: #5b6b82; font-size: 0.92rem; }}
    @media (max-width: 1100px) {{
      .subgrid, .four-up {{ grid-template-columns: repeat(2, 1fr); }}
    }}
    @media (max-width: 820px) {{
      .grid, .subgrid, .four-up, .verdict {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>QAA Quantum Exploration Dashboard</h1>
      <p>This dashboard compares baseline search performance and the new Quantum Tree distillation architecture for the 3D two-aircraft separation scenario. Monte Carlo and QAOA remain directly comparable by iterations, while Quantum Tree is shown as a teacher-quantum-student training pipeline.</p>
      <div class="tag">{html.escape(comparison['question'])}</div>{quantum_tag}
      <div class="verdict">
        <div class="card"><h3>Research verdict</h3><div>{html.escape(verdict)}</div></div>
        <div class="card"><h3>AUC advantage</h3><div>{summary['auc_advantage_pct']}%</div></div>
        <div class="card"><h3>Time to first failure</h3><div>QAOA {summary['qaoa_time_to_first_failure']} / MC {summary['mc_time_to_first_failure']}</div></div>
      </div>
    </section>

    <section class="grid">
      <div class="card">
        <h2>Panel 1 - Cumulative failures over K iterations</h2>
        {_line_chart_svg(
            comparison['per_iteration']['qaoa'],
            comparison['per_iteration']['monte_carlo'],
            qtrd_vs_mc['per_iteration']['qtrd'] if qtrd_vs_mc is not None else None,
        )}
      </div>
      <div class="card">
        <h2>Panel 2 - Failure type breakdown</h2>
        {_bar_chart_svg(summary)}
        <div class="small">SL = separation loss, NM = near miss</div>
      </div>
    </section>

    <section class="subgrid">
      <div class="card stats">
        <h2>Panel 3 - Summary statistics</h2>
        <p><span>QAOA unique failures</span><strong>{summary['qaoa_unique_failures']}</strong></p>
        <p><span>MC unique failures</span><strong>{summary['mc_unique_failures']}</strong></p>
        <p><span>QTRD unique failures</span><strong>{qtrd_vs_mc['summary']['qtrd_unique_failures'] if qtrd_vs_mc is not None else 'n/a'}</strong></p>
        <p><span>QAOA AUC</span><strong>{summary['qaoa_auc']}</strong></p>
        <p><span>MC AUC</span><strong>{summary['mc_auc']}</strong></p>
        <p><span>QTRD AUC</span><strong>{qtrd_vs_mc['summary']['qtrd_auc'] if qtrd_vs_mc is not None else 'n/a'}</strong></p>
        <p><span>Mean unique failures</span><strong>{summary['mean_unique_failures']}</strong></p>
        <p><span>Std unique failures</span><strong>{summary['std_unique_failures']}</strong></p>
      </div>
      <div class="card stats">
        <h2>Panel 4 - Circuit summary</h2>
        <p><span>Backend</span><strong>{html.escape(comparison['qaoa_circuit_info']['backend'])}</strong></p>
        <p><span>Qubits</span><strong>{comparison['qaoa_circuit_info']['num_qubits']}</strong></p>
        <p><span>Depth</span><strong>{comparison['qaoa_circuit_info']['depth']}</strong></p>
        <p><span>Parameters</span><strong>{comparison['qaoa_circuit_info']['num_params']}</strong></p>
        <p><span>Reps</span><strong>{comparison['qaoa_circuit_info']['reps']}</strong></p>
      </div>
      <div class="card">
        <h2>Panel 5 - QAOA ansatz diagram</h2>
        <pre>{html.escape(comparison['circuit_text'])}</pre>
      </div>
    </section>

    {f'''
    <section class="card full">
      <h2>Quantum Tree vs Monte Carlo</h2>
      <div class="small">{html.escape(qtrd_vs_mc["question"])}</div>
      <div class="subgrid" style="margin-top:18px;">
        <div class="card stats">
          <h3>Direct verdict</h3>
          <p><span>Winner</span><strong>{html.escape(qtrd_vs_mc["winner"])}</strong></p>
          <p><span>QTRD unique failures</span><strong>{qtrd_vs_mc["summary"]["qtrd_unique_failures"]}</strong></p>
          <p><span>MC unique failures</span><strong>{qtrd_vs_mc["summary"]["mc_unique_failures"]}</strong></p>
          <p><span>QTRD TTFF</span><strong>{qtrd_vs_mc["summary"]["qtrd_time_to_first_failure"]}</strong></p>
          <p><span>MC TTFF</span><strong>{qtrd_vs_mc["summary"]["mc_time_to_first_failure"]}</strong></p>
        </div>
        <div class="card stats">
          <h3>Model metrics</h3>
          <p><span>Teacher accuracy</span><strong>{qtrd_vs_mc["summary"]["teacher_accuracy"]}</strong></p>
          <p><span>Student MSE</span><strong>{qtrd_vs_mc["summary"]["student_mse"]}</strong></p>
          <p><span>Backend</span><strong>{html.escape(qtrd_vs_mc["summary"]["quantum_backend"])}</strong></p>
          <p><span>AUC advantage</span><strong>{qtrd_vs_mc["summary"]["auc_advantage_pct"]}%</strong></p>
        </div>
        <div class="card stats">
          <h3>Failure mix</h3>
          <p><span>QTRD separation loss</span><strong>{qtrd_vs_mc["summary"]["qtrd_failure_types"]["separation_loss"]}</strong></p>
          <p><span>QTRD near miss</span><strong>{qtrd_vs_mc["summary"]["qtrd_failure_types"]["near_miss"]}</strong></p>
          <p><span>MC separation loss</span><strong>{qtrd_vs_mc["summary"]["mc_failure_types"]["separation_loss"]}</strong></p>
          <p><span>MC near miss</span><strong>{qtrd_vs_mc["summary"]["mc_failure_types"]["near_miss"]}</strong></p>
        </div>
      </div>
    </section>
    ''' if qtrd_vs_mc is not None else ''}

    <section class="card full">
      <h2>Top 10 failure-inducing initial states</h2>
      <table>
        <thead>
          <tr>
            <th>#</th><th>Heading</th><th>Altitude</th><th>Speed</th><th>X offset</th><th>Min H sep</th><th>Min V sep</th><th>Type</th>
          </tr>
        </thead>
        <tbody>{_table_rows(comparison['top_failures'])}</tbody>
      </table>
    </section>
{quantum_section}
  </div>
</body>
</html>"""

    dashboard_path = OUT / "dashboard.html"
    dashboard_path.write_text(html_doc, encoding="utf-8")
    return dashboard_path


def main() -> None:
    dashboard = render_dashboard()
    print(f"Dashboard written: {dashboard}")


if __name__ == "__main__":
    main()
