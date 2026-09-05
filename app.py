from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Chat with your documents",
    page_icon="📚",
    layout="wide",
)

SUPPORTED_FORMATS = ["txt", "md", "pdf"]


def _pick_config() -> object:
    from rag.config import load_config

    cfg_file = os.getenv("RAG_CONFIG", "config.yaml")
    plan = load_config(cfg_file)
    return plan.experiments[0]


def _load_plan() -> object:
    from rag.config import load_config

    return load_config(os.getenv("RAG_CONFIG", "config.yaml"))


def _run_benchmark_worker(plan, exp_names: list[str], metrics: list[str]) -> None:
    """Runs the RAGAS benchmark in a background thread, appending per-question
    rows to a shared dict so the UI can show progress live."""
    from benchmark.store import ResultsStore
    from evaluation.ragas_eval import run_ragas_evaluation
    from evaluation.testset import load_test_set
    from rag.loader import load_documents
    from rag.pipeline import RAGPipeline

    bench = st.session_state["bench"]
    bench["status"] = "Loading documents & test questions..."
    try:
        documents = load_documents(plan.docs_dir)
        test_questions = load_test_set(plan.test_set)
        experiments = [e for e in plan.experiments if e.name in exp_names]
        total = len(experiments) * len(test_questions)
        done = 0

        for exp in experiments:
            bench["status"] = f"Indexing + answering with {exp.name}..."
            pipeline = RAGPipeline(exp, documents)
            for q in test_questions:
                result = pipeline.answer(q["user_input"], q.get("reference", ""), q.get("reference_contexts"))
                sample = [
                    {
                        "user_input": q["user_input"],
                        "retrieved_contexts": result.contexts,
                        "response": result.answer,
                        "reference": result.reference,
                    }
                ]
                agg, per = run_ragas_evaluation(
                    sample,
                    metrics,
                    exp.judge_llm,
                    exp.embedding_model,
                    exp.provider,
                    exp.base_url,
                    exp.embedding_backend,
                )
                row: dict[str, object] = {"experiment": exp.name, "question": q["user_input"]}
                if per:
                    row.update({k: v for k, v in per[0].items() if k not in ("user_input", "response")})
                bench["rows"].append(row)
                done += 1
                bench["status"] = f"{exp.name}: {done}/{total} questions scored..."
        bench["done_experiments"] = exp_names

        try:
            store = ResultsStore(plan.results_dir / "benchmark.db")
            for exp in experiments:
                exp_rows = [r for r in bench["rows"] if r["experiment"] == exp.name]
                store.insert(exp.name, exp.to_dict(), _aggregate(exp_rows, metrics))
            store.close()

            per_q = pd.DataFrame(bench["rows"])
            per_q.to_csv(plan.results_dir / "per_question.csv", index=False, encoding="utf-8")
            summary = pd.DataFrame(
                [
                    {"experiment": exp.name, "config": exp.to_dict(), **_aggregate(
                        [r for r in bench["rows"] if r["experiment"] == exp.name], metrics)}
                    for exp in experiments
                ]
            )
            summary.to_csv(plan.results_dir / "results.csv", index=False, encoding="utf-8")
            bench["saved"] = True
        except Exception:
            bench["saved"] = False  # read-only cloud mount -> keep results in-memory only

        bench["status"] = "Done."
        bench["done"] = True
    except Exception as exc:  # pragma: no cover - surfaces in the UI
        bench["error"] = str(exc)
        bench["status"] = f"Failed: {exc}"
        bench["done"] = True


def _aggregate(rows: list[dict], metrics: list[str]) -> dict[str, float]:
    import statistics

    agg: dict[str, float] = {}
    for m in metrics:
        vals = [r[m] for r in rows if m in r and r[m] is not None and r[m] == r[m]]
        if vals:
            agg[m] = round(statistics.mean(vals), 4)
    return agg


def _build_index(uploaded_files: list) -> str:
    from langchain_core.documents import Document

    from rag.loader import SUPPORTED_EXTS
    from rag.pipeline import RAGPipeline

    exp = st.session_state.config
    tmp = Path(st.session_state.tmp_dir)
    files = []
    for f in uploaded_files:
        suffix = Path(f.name).suffix.lower()
        if suffix not in SUPPORTED_EXTS:
            continue
        dest = tmp / f.name
        dest.write_bytes(f.getvalue())
        files.append(dest)

    if not files:
        return "no_files"

    documents: list[Document] = []
    from rag.loader import _load_pdf, load_documents

    supported = [p for p in tmp.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]
    for path in supported:
        if path.suffix.lower() == ".pdf":
            documents.extend(_load_pdf(path))
        else:
            from langchain_community.document_loaders import TextLoader

            loader = TextLoader(str(path), encoding="utf-8")
            for doc in loader.load():
                doc.metadata = {"source": path.name}
                documents.append(doc)

    if not documents:
        return "no_files"

    st.session_state.pipeline = RAGPipeline(exp, documents)
    return "ok"


def _render_sources(result) -> None:
    with st.expander(f"Retrieved sources ({len(result.sources)})"):
        for source, context in zip(result.sources, result.contexts):
            st.markdown(f"**{source}**")
            st.text(context[:600] + ("..." if len(context) > 600 else ""))
            st.divider()


def _handle_question() -> None:
    question = st.session_state.get("chat_input", "").strip()
    if not question:
        return
    st.session_state.history.append({"role": "user", "content": question})
    pipeline = st.session_state.get("pipeline", None)
    if pipeline is None:
        st.session_state.history.append(
            {"role": "assistant", "content": "⚠️ Please upload some documents and build an index first."}
        )
        return
    with st.spinner("Searching your documents and writing an answer (can take 30-60s)..."):
        try:
            result = pipeline.answer(question)
            text = result.answer or "No answer produced."
            if result.sources:
                text += f"\n\n📎 Retrieved from: {', '.join(sorted(set(result.sources)))}"
            st.session_state.history.append({"role": "assistant", "content": text})
        except Exception as exc:
            st.session_state.history.append(
                {"role": "assistant", "content": f"⚠️ Error: {exc}\n\nIs Ollama running? (start the Ollama app)"}
            )


def main() -> None:
    st.title("📚 Chat with your documents")
    st.caption(
        "Upload your own .txt / .md / .pdf files, then ask questions. Answers come only from "
        "your documents - powered by a free RAG pipeline (Groq + FastEmbed on the cloud, "
        "or Ollama locally). Check the 📊 Benchmark tab for accuracy scores."
    )

    if "config" not in st.session_state:
        st.session_state.config = _pick_config()
    if "plan" not in st.session_state:
        st.session_state.plan = _load_plan()
    if "results_dir" not in st.session_state:
        st.session_state.results_dir = st.session_state.plan.results_dir
    if "tmp_dir" not in st.session_state:
        st.session_state.tmp_dir = tempfile.mkdtemp(prefix="ragdocs_")
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "bench_running" not in st.session_state:
        st.session_state.bench_running = False
    if "bench" not in st.session_state:
        st.session_state.bench = {"rows": [], "done_experiments": [], "done": False, "status": ""}

    tab_chat, tab_bench, tab_scores = st.tabs(
        ["💬 Chat", "▶ Run benchmark", "📊 Benchmark scores"]
    )

    with tab_chat:
        try:
            _render_chat_tab()
        except Exception as exc:
            st.error(f"⚠️ Something went wrong: {exc}")
            st.caption("If this happened while building the index, check the app logs "
                       "(Manage app -> Logs). On Streamlit Cloud the first build also "
                       "downloads the embedding model (~30s).")

    with tab_bench:
        _render_bench_tab(st.session_state.plan)

    with tab_scores:
        from scores_view import render_scores

        bench = st.session_state.bench
        render_scores(
            st.session_state.results_dir,
            session_runs=_session_runs_from_bench(bench),
            session_per_q=pd.DataFrame(bench["rows"]) if bench["rows"] else None,
        )


def _session_runs_from_bench(bench: dict):
    if not bench["rows"] or not bench.get("done"):
        return None
    metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    rows = []
    for exp_name in dict.fromkeys(r["experiment"] for r in bench["rows"]):
        exp_rows = [r for r in bench["rows"] if r["experiment"] == exp_name]
        row = {"experiment": exp_name, **_aggregate(exp_rows, metrics)}
        rows.append(row)
    return pd.DataFrame(rows)


def _render_bench_tab(plan) -> None:
    st.subheader("⏱️ Run a benchmark from here")
    st.caption(
        "Answers each question in `test_set.json` with every selected experiment and "
        "scores them with RAGAS (judge LLM = Groq on the cloud, Ollama locally). "
        "Results appear live, question by question."
    )

    exp_names = [e.name for e in plan.experiments]
    selected = st.multiselect(
        "Experiments to run", exp_names, default=[exp_names[0]], key="bench_exp_select"
    )
    col_note, col_btn = st.columns([3, 1])
    with col_note:
        st.caption(
            f"Questions: {len(load_test_set_count(plan))} per experiment. "
            "On the free Groq tier keep it to 1 experiment (≈40 LLM calls) to stay in quota."
        )
    with col_btn:
        start = st.button("▶ Run benchmark", type="primary", use_container_width=True)

    if start and not st.session_state.bench_running:
        bench = {"rows": [], "done_experiments": [], "done": False, "status": "Starting...",
                 "saved": None, "error": None}
        st.session_state.bench = bench
        st.session_state.bench_running = True
        st.session_state.config = _pick_config()
        thread = threading.Thread(
            target=_run_benchmark_worker,
            args=(plan, selected, st.session_state.config.metrics),
            daemon=True,
        )
        thread.start()
        st.rerun()

    bench = st.session_state.bench

    with st.container(border=True):
        if st.session_state.bench_running and not bench.get("done"):
            st.info(f"Running… {bench.get('status', '')}  \n(you can switch tabs, progress keeps updating)")
        elif bench.get("done"):
            if bench.get("error"):
                st.error(f"Benchmark failed: {bench['error']}")
            else:
                if bench.get("saved"):
                    st.success("Benchmark done — saved to `results/benchmark.db`.")
                else:
                    st.info("Benchmark done — showing results in this session (cloud files are read-only).")
        else:
            st.caption("Nothing run yet — pick experiments and press ▶ Run benchmark.")

    with st.container(border=True):
        st.markdown("**Progress — per-question scores**")
        if bench["rows"]:
            st.dataframe(pd.DataFrame(bench["rows"]), use_container_width=True, hide_index=True)
        else:
            st.caption("_no questions scored yet_")

    if bench.get("done") and bench["rows"] and not bench.get("error"):
        runs = _session_runs_from_bench(bench)
        if runs is not None:
            st.subheader("Aggregated scores")
            st.dataframe(runs, use_container_width=True, hide_index=True)


def load_test_set_count(plan) -> int:
    from evaluation.testset import load_test_set

    return len(load_test_set(plan.test_set))


def _render_chat_tab() -> None:
    with st.sidebar:
        st.header("1️⃣ Upload documents")
        uploaded = st.file_uploader(
            "Drop your files here",
            type=SUPPORTED_FORMATS,
            accept_multiple_files=True,
            key="uploader",
        )
        if st.button("🔨 Build index", type="primary", use_container_width=True):
            if not uploaded:
                st.warning("Upload at least one file first.")
            else:
                with st.spinner("Building the index... (can take a while)"):
                    status = _build_index(uploaded)
                if status == "no_files":
                    st.error("No supported files found. Use .txt, .md or .pdf.")
                else:
                    st.session_state.history = []
                    st.success(
                        f"Index ready! {len(uploaded)} file(s) indexed. Ask below.",
                        icon="✅",
                    )
        if st.button("🗑️ Reset session", use_container_width=True):
            for key in ("pipeline", "history", "tmp_dir", "config"):
                st.session_state.pop(key, None)
            st.rerun()

        st.divider()
        st.header("ℹ️ About")
        st.caption(
            f"Model: **{st.session_state.config.llm}**  \n"
            f"Embeddings: **{st.session_state.config.embedding_model}**  \n"
            f"Provider: **{st.session_state.config.provider}**  \n"
            f"Chunk size: {st.session_state.config.chunk_size}  •  "
            f"overlap: {st.session_state.config.chunk_overlap}  •  "
            f"top-k: {st.session_state.config.top_k}"
        )
        st.caption(
            "Answers are generated ONLY from your uploaded documents. If the answer "
            "isn't in your files, the assistant says so."
        )

    st.header("2️⃣ Ask a question")
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    st.chat_input("Ask about your documents...", key="chat_input", on_submit=_handle_question)


if __name__ == "__main__":
    main()