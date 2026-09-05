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

st.set_page_config(
    page_title="RAG Benchmark Dashboard",
    page_icon="📊",
    layout="wide",
)


def _load_history(db_path: Path) -> pd.DataFrame:
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(
        "SELECT id, created_at, experiment, context_precision, context_recall, "
        "faithfulness, answer_relevancy FROM runs ORDER BY id DESC",
        conn,
    )
    conn.close()
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


def main() -> None:
    st.title("📊 RAG Benchmark Dashboard")
    st.caption(
        "Scores from `results/benchmark.db` - run `python run_benchmark.py` after any "
        "config change, then refresh this page. Higher is better (0-1)."
    )

    db_path = Path("results/benchmark.db")
    if not db_path.exists():
        st.warning("No results yet. Run `python run_benchmark.py --smoke` or a full run first.")
        return

    df = _load_history(db_path)
    if df.empty:
        st.warning("No runs recorded yet. Run `python run_benchmark.py` first.")
        return

    recent = _latest_per_experiment(df)

    metric_cols = [c for c in df.columns if c in METRIC_INFO]

    st.subheader("Latest run per experiment")
    st.dataframe(
        recent.style.format({c: "{:.4f}" for c in metric_cols}, na_rep="-"),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Scores at a glance")
    cols = st.columns(len(metric_cols))
    for col, box in zip(metric_cols, cols):
        vals = recent[col].dropna()
        with box:
            st.metric(
                label=col.replace("_", " ").title(),
                value=f"{vals.mean():.3f}" if not vals.empty else "-",
                help=METRIC_INFO[col],
            )

    st.subheader("Charts")
    chart_cols = st.columns(2)
    for i, col in enumerate(metric_cols):
        with chart_cols[i % 2]:
            st.markdown(f"**{col.replace('_', ' ').title()}**")
            st.caption(METRIC_INFO[col])
            _metric_bar_chart(recent, col)

    st.subheader("All recorded runs")
    show = df.copy()
    for c in metric_cols:
        show[c] = show[c].round(4)
    st.dataframe(show, use_container_width=True, hide_index=True)

    csv_path = Path("results/per_question.csv")
    if csv_path.exists():
        st.subheader("Per-question detail")
        per_q = pd.read_csv(csv_path)
        exp_name = st.selectbox("Experiment", sorted(per_q["experiment"].astype(str).unique()))
        st.dataframe(
            per_q[per_q["experiment"] == exp_name],
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Tip: a deliberately 'unanswerable' question is in the test set - if the model "
        "still answers it confidently, `faithfulness` will expose the hallucination."
    )


if __name__ == "__main__":
    main()