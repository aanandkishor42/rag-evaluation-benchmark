from __future__ import annotations

import html as _html
from typing import Any, Sequence

METRIC_LABELS = {
    "context_precision": "Context Precision - how relevant were the retrieved chunks?",
    "context_recall": "Context Recall - did retrieval find the correct info at all?",
    "faithfulness": "Faithfulness - does the answer stick to the docs (no hallucination)?",
    "answer_relevancy": "Answer Relevancy - is the answer actually about the question?",
}


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{value:0.4f}" if isinstance(value, float) else str(value)


def _render_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)
    ]
    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for row in rows:
        print(" | ".join(cell.ljust(w) for cell, w in zip(row, widths)))


def print_comparison(runs: Sequence[dict[str, Any]]) -> None:
    """Print the aggregate-score comparison table for this run."""
    if not runs:
        print("No runs to report.")
        return
    metrics = _collect_metrics(runs)
    headers = ["experiment", *metrics]
    rows = [
        [run["experiment"], *[_fmt(run["scores"].get(m)) for m in metrics]]
        for run in runs
    ]
    print("\n=== Benchmark comparison ===")
    _render_table(headers, rows)
    print()


def print_history(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("\nNo previous runs recorded yet.")
        return
    metric_cols = [k for k in rows[0] if k not in ("id", "created_at", "experiment")]
    headers = ["id", "created_at", "experiment", *metric_cols]
    table_rows = [
        [
            str(r["id"]),
            r["created_at"][:21],
            str(r["experiment"]),
            *[_fmt(r.get(m)) for m in metric_cols],
        ]
        for r in rows
    ]
    print("\n=== Run history (most recent first) ===")
    _render_table(headers, table_rows)
    print()


def _collect_metrics(runs: Sequence[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for run in runs:
        for name in run.get("scores", {}):
            if name not in seen:
                seen.append(name)
    return seen


def _latest_per_experiment(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the most recent row for each experiment name."""
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:  # history() is newest-first
        name = row["experiment"]
        if name not in latest:
            latest[name] = row
    return list(latest.values())


def _bar(value: float | None, width_px: int = 260) -> str:
    if value is None:
        return '<div class="bar none"><span>no data</span></div>'
    pct = max(2.0, value * 100.0)
    return (
        f'<div class="bar"><div class="bar-fill" style="width:{pct:.0f}%;"></div>'
        f'<span class="bar-val">{value:0.3f}</span></div>'
    )


def write_html_report(history_rows: list[dict[str, Any]], out_path: Any) -> str:
    """Write a self-contained HTML report (no internet needed) and return its path."""
    runs = _latest_per_experiment(history_rows)
    metric_cols = [k for k in history_rows[0] if k not in ("id", "created_at", "experiment")] if history_rows else []

    table_body = "".join(
        "<tr>"
        f"<td class='exp'>{_html.escape(str(r['experiment']))}</td>"
        + "".join(
            f"<td class='sc'>{_fmt(r.get(m))}</td>" for m in metric_cols
        )
        + f"<td class='ts'>{str(r['created_at'])[:19].replace('T', ' ')}</td>"
        + "</tr>"
        for r in runs
    )
    headers = "".join(
        f"<th>{_html.escape(m.replace('_', ' '))}</th>" for m in metric_cols
    )

    charts = ""
    for m in metric_cols:
        tip = METRIC_LABELS.get(m, "")
        rows_html = "".join(
            f"<div class='chart-row'><span class='chart-lab'>{_html.escape(str(r['experiment']))}</span>"
            f"{_bar(r.get(m))}</div>"
            for r in runs
        )
        charts += (
            f"<div class='chart-card'><h3>{_html.escape(m.replace('_', ' '))}</h3>"
            f"<p class='tip'>{_html.escape(tip)}</p>{rows_html}</div>"
        )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RAG Benchmark Report</title>
<style>
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; background: #f4f6f9; color: #1c2333; }}
  header {{ background: #1e2a4a; color: #fff; padding: 28px 40px; }}
  header h1 {{ margin: 0; font-size: 22px; }}
  header p {{ margin: 6px 0 0; opacity: .8; font-size: 13px; }}
  main {{ padding: 28px 40px; max-width: 1000px; margin: 0 auto; }}
  h2 {{ font-size: 16px; margin: 8px 0 14px; }}
  .cards {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px 20px; flex: 1 1 220px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
  .card .big {{ font-size: 30px; font-weight: 700; }}
  .card .lbl {{ font-size: 12px; color: #64748b; margin-top: 2px; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; overflow: hidden; }}
  th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #eef1f6; font-size: 13px; }}
  th {{ background: #eef1f6; text-transform: capitalize; font-size: 12px; color: #475569; }}
  td.exp {{ font-weight: 600; }}
  td.sc {{ font-family: Consolas, monospace; }}
  td.ts {{ color: #94a3b8; font-size: 12px; }}
  .blk {{ display: block; height: 16px; margin: 18px 0; }}
  .chart-card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px 20px; margin: 14px 0; }}
  .chart-card h3 {{ margin: 0 0 2px; font-size: 14px; text-transform: capitalize; }}
  .chart-card .tip {{ margin: 0 0 12px; font-size: 12px; color: #64748b; }}
  .chart-row {{ display: flex; align-items: center; gap: 10px; margin: 7px 0; }}
  .chart-lab {{ width: 170px; font-size: 12px; font-weight: 600; }}
  .bar {{ position: relative; flex: 1; max-width: 360px; height: 18px; background: #eef1f6; border-radius: 9px; }}
  .bar-fill {{ height: 100%; background: linear-gradient(90deg, #3b82f6, #06b6d4); border-radius: 9px; }}
  .bar-val {{ position: absolute; right: 8px; top: 1px; font-size: 11px; color: #1c2333; font-weight: 600; }}
  .bar.none {{ display:flex; align-items:center; justify-content:center; color:#94a3b8; font-size:11px; }}
  footer {{ padding: 14px 40px 34px; font-size: 12px; color: #94a3b8; }}
</style>
</head>
<body>
<header>
  <h1>RAG Evaluation &amp; Benchmarking Pipeline</h1>
  <p>Generated from results/benchmark.db &middot; no internet connection required</p>
</header>
<main>
  <h2>Latest run per experiment</h2>
  <table>
    <tr><th>experiment</th>{headers}<th>run time (UTC)</th></tr>
    {table_body}
  </table>
  <div class="blk"></div>
  <h2>Score charts</h2>
  {charts}
</main>
<footer>Tip: run `python run_benchmark.py` after any config change to refresh the scores here. Higher is better (0-1).</footer>
</body>
</html>"""
    out_path.write_text(page, encoding="utf-8")
    return str(out_path)