from __future__ import annotations

import streamlit as st

from scores_view import render_scores

st.set_page_config(
    page_title="RAG Benchmark Dashboard",
    page_icon="📊",
    layout="wide",
)


def main() -> None:
    st.title("📊 RAG Benchmark Dashboard")
    st.caption(
        "Scores from benchmark runs - run `python run_benchmark.py` after any config "
        "change, then refresh this page. Higher is better (0-1)."
    )

    render_scores("results")


if __name__ == "__main__":
    main()