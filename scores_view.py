from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

METRIC_INFO = {
    "context_precision": "of the chunks retrieved, how many were actually relevant to the question?",
    "context_recall": "did retrieval surface the correct information from the docs at all?",
    "faithfulness": "does every claim in the answer trace back to the retrieved chunks (no hallucination)?",
    "answer_relevancy": "does the answer actually address the question asked?",
}


def _load_history(db_path: Path) -> pd.DataFrame:
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(
        "SELECT id, created_at, experiment, context_precision, context_recall, "
        "faithfulness, answer_relevancy FROM runs ORDER BY id DESC",
        conn,
    )
    conn.close()
    for c in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _latest_per_experiment(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("created_at", ascending=False).drop_duplicates("experiment")


def _metric_bar_chart(df: pd.DataFrame, col: str) -> None:
    small = df[df[col].notna()][["experiment", col]].copy()
    if small.empty:
        st.write("_no scores recorded for this metric yet_")
        return
    small = small.sort_values(col)
    st.bar_chart(small.set_index("experiment"), color=col, height=300)


def render_scores(
    results_dir: str | Path = "results",
    session_runs: pd.DataFrame | None = None,
    session_per_q: pd.DataFrame | None = None,
) -> None:
    """Render the benchmark scoreboard. Priority: in-memory run from this session,
    then local results_dir (gitignored), then committed sample_results/ so the
    cloud app still shows scores."""
    if session_runs is not None and not session_runs.empty:
        _render_session(session_runs, session_per_q)
        return

    db_path = Path(results_dir) / "benchmark.db"
    if db_path.exists():
        _render_from_db(db_path)
        return

    sample_csv = Path("sample_results/benchmark_results.csv")
    if sample_csv.exists():
        _render_from_csv(sample_csv, Path("sample_results/per_question.csv"))
        return

    st.warning(
        "No scores yet. Run a benchmark from the ▶ Run benchmark tab, or "
        "`python run_benchmark.py` on your machine."
    )


def _render_session(runs: pd.DataFrame, per_q: pd.DataFrame | None) -> None:
    metric_cols = [c for c in runs.columns if c in METRIC_INFO]
    st.caption("Showing the benchmark you just ran in this session (not yet saved to disk).")
    st.markdown("## Latest run per experiment")
    st.dataframe(
        runs.style.format({c: "{:.4f}" for c in metric_cols}, na_rep="-"),
        use_container_width=True,
        hide_index=True,
    )
    _render_glance_and_charts(runs, metric_cols)
    if per_q is not None and not per_q.empty and "experiment" in per_q.columns:
        exp_name = st.selectbox("Per-question detail", sorted(per_q["experiment"].astype(str).unique()))
        st.dataframe(per_q[per_q["experiment"] == exp_name], use_container_width=True, hide_index=True)


def _render_from_db(db_path: Path) -> None:
    df = _load_history(db_path)
    if df.empty:
        st.warning("No runs recorded yet. Run `python run_benchmark.py` first.")
        return
    recent = _latest_per_experiment(df)
    _render_recent(recent)

    csv_path = db_path.parent / "per_question.csv"
    if csv_path.exists():
        _render_per_question(csv_path)


def _render_from_csv(results_csv: Path, per_question_csv: Path | None) -> None:
    df = pd.read_csv(results_csv)
    metric_cols = [c for c in df.columns if c in METRIC_INFO]
    if df.empty:
        st.warning("Sample results file is empty.")
        return
    recent = df.copy()
    st.caption("Showing committed sample results (from a local Ollama run).")
    st.markdown("## Latest run per experiment")
    st.dataframe(
        recent.style.format({c: "{:.4f}" for c in metric_cols}, na_rep="-"),
        use_container_width=True,
        hide_index=True,
    )
    _render_glance_and_charts(recent, metric_cols)
    st.markdown("## All recorded runs")
    st.dataframe(recent, use_container_width=True, hide_index=True)

    if per_question_csv is not None and per_question_csv.exists():
        _render_per_question(per_question_csv)


def _render_recent(recent: pd.DataFrame) -> None:
    metric_cols = [c for c in recent.columns if c in METRIC_INFO]
    if "created_at" in recent.columns:
        recent = recent.drop(columns=["created_at"])

    st.markdown("## Latest run per experiment")
    st.dataframe(
        recent.style.format({c: "{:.4f}" for c in metric_cols}, na_rep="-"),
        use_container_width=True,
        hide_index=True,
    )
    _render_glance_and_charts(recent, metric_cols)

    st.markdown("## All recorded runs")
    show = recent.copy()
    for c in metric_cols:
        show[c] = pd.to_numeric(show[c], errors="coerce").round(4)
    st.dataframe(show, use_container_width=True, hide_index=True)

    full_blank = show[metric_cols].isna().all(axis=1)
    if full_blank.any():
        st.caption(
            f"{int(full_blank.sum())} run(s) had no scorable scores (judge couldn't "
            "score them — check Groq quota / per-question `status` column)."
        )


def _render_glance_and_charts(recent: pd.DataFrame, metric_cols: list[str]) -> None:
    if not metric_cols:
        st.caption(
            "No metric values in this run — the judge couldn't score these questions "
            "(check Groq quota / logs, or the answers were refusals)."
        )
        return
    st.markdown("## Scores at a glance")
    cols = st.columns(len(metric_cols))
    for col, box in zip(metric_cols, cols):
        vals = recent[col].dropna()
        with box:
            st.metric(
                label=col.replace("_", " ").title(),
                value=f"{vals.mean():.3f}" if not vals.empty else "-",
                help=METRIC_INFO[col],
            )

    st.markdown("## Charts")
    chart_cols = st.columns(2)
    for i, col in enumerate(metric_cols):
        with chart_cols[i % 2]:
            st.markdown(f"**{col.replace('_', ' ').title()}**")
            st.caption(METRIC_INFO[col])
            _metric_bar_chart(recent, col)


def _render_per_question(csv_path: Path) -> None:
    per_q = pd.read_csv(csv_path)
    st.markdown("## Per-question detail")
    if "experiment" in per_q.columns:
        exp_name = st.selectbox("Experiment", sorted(per_q["experiment"].astype(str).unique()))
        per_q = per_q[per_q["experiment"] == exp_name]
    st.dataframe(per_q, use_container_width=True, hide_index=True)

    st.caption(
        "Tip: a deliberately 'unanswerable' question is in the test set - if the model "
        "still answers it confidently, `faithfulness` will expose the hallucination."
    )