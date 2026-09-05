from __future__ import annotations

import io
import json
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


def _run_benchmark_worker(
    plan, exp_names: list[str], metrics: list[str], custom_questions=None, documents=None
) -> None:
    """Runs the RAGAS benchmark in a background thread, appending per-question
    rows to a shared dict so the UI can show progress live."""
    bench = st.session_state["bench"]
    bench["status"] = "Warming up (importing libraries + embedding model)…"
    try:
        from benchmark.store import ResultsStore
        from evaluation.ragas_eval import run_ragas_evaluation
        from evaluation.testset import load_test_set
        from rag.loader import load_documents
        from rag.pipeline import RAGPipeline

        bench["status"] = "Loading documents & test questions..."
        documents = documents if documents is not None else load_documents(plan.docs_dir)
        test_questions = custom_questions or load_test_set(plan.test_set)
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
                    for k, v in per[0].items():
                        if k in ("user_input", "response"):
                            continue
                        row[k] = round(v, 4) if isinstance(v, float) else v
                bench["rows"].append(row)
                done += 1
                bench["status"] = f"{exp.name}: {done}/{total} questions scored..."
        bench["done_experiments"] = exp_names
        bench["source"] = "custom" if custom_questions else "test_set.json"

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
    from rag.pipeline import RAGPipeline

    tmp = Path(st.session_state.tmp_dir)
    documents = _build_documents(uploaded_files, tmp)
    if not documents:
        return "no_files"
    st.session_state.pipeline = RAGPipeline(st.session_state.config, documents)
    return "ok"


def _build_documents(uploaded_files: list, tmp_dir: Path) -> list:
    from langchain_community.document_loaders import TextLoader
    from langchain_core.documents import Document

    from rag.loader import SUPPORTED_EXTS, _load_pdf

    saved = []
    for f in uploaded_files:
        suffix = Path(f.name).suffix.lower()
        if suffix not in SUPPORTED_EXTS:
            continue
        dest = tmp_dir / f.name
        dest.write_bytes(f.getvalue())
        saved.append(dest)
    if not saved:
        return []

    documents: list[Document] = []
    for path in tmp_dir.rglob("*"):
        if path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        if path.suffix.lower() == ".pdf":
            documents.extend(_load_pdf(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8")
            for doc in loader.load():
                doc.metadata = {"source": path.name}
                documents.append(doc)
    return documents


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
        st.session_state.bench = {
            "rows": [], "done_experiments": [], "done": False, "status": "",
            "saved": None, "error": None, "source": None,
        }

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
        "Answers each question with every selected experiment and scores it with RAGAS "
        "(judge LLM = Groq on the cloud, Ollama locally). Results appear live, question "
        "by question. Use the bundled test questions OR upload your own."
    )
    st.markdown(
        "**How to read the scores:**  \n"
        "• **faithfulness** = the answer stays true to the docs (the hallucination check).  \n"
        "• **context_recall** = the right chunk was found (needs a `reference` / expected answer).  \n"
        "• **context_precision** = of the chunks shown, how many were relevant (needs the exact "
        "supporting passage — often empty for imported tests).  \n"
        "• **answer_relevancy** = the answer actually addresses the question asked."
    )

    exp_names = [e.name for e in plan.experiments]
    selected = st.multiselect(
        "Experiments to run", exp_names, default=[exp_names[0]], key="bench_exp_select"
    )

    st.markdown("**1️⃣ Documents to benchmark against**")
    docs_choice = st.radio(
        "Choose the source documents for this run:",
        ["Bundled sample docs", "Upload my own documents"],
        horizontal=True,
        key="bench_docs_choice",
    )
    docs_for_run = None
    if docs_choice == "Upload my own documents":
        files = st.file_uploader(
            "Upload your docs (.txt / .md / .pdf)", type=SUPPORTED_FORMATS,
            accept_multiple_files=True, key="bench_docs",
        )
        if files:
            tmpd = Path(tempfile.mkdtemp(prefix="benchdocs_"))
            docs_for_run = _build_documents(files, tmpd)
            if docs_for_run:
                st.success(f"{len(docs_for_run)} document(s) loaded for this benchmark run.")
            else:
                st.warning("No supported files found. Use .txt, .md or .pdf.")
        else:
            st.caption("_no files chosen yet_")

    st.markdown("**2️⃣ Questions to benchmark**")
    st.caption(
        "These must be **your** questions about your own document. Either upload a CSV "
        "(`question,reference`) or just type them below. `reference` = the expected answer "
        "you take from your document (it powers `context_recall`)."
    )
    custom_q = _render_test_set_uploader()
    q_source = "uploaded file"
    if not custom_q:
        custom_q = _questions_from_textareas()
        q_source = "typed questions"
    st.download_button(
        "📎 Template CSV (blank — fill in YOUR questions)",
        data=_template_csv(plan, custom_q),
        file_name="template.csv",
        mime="text/csv",
        use_container_width=False,
    )
    if custom_q:
        preview = pd.DataFrame(
            {
                "question": [q["user_input"] for q in custom_q],
                "expected answer (reference)": [q.get("reference", "") for q in custom_q],
            }
        )
        st.success(f"Using your {q_source}: {len(custom_q)} questions.")
        st.dataframe(preview, use_container_width=True, hide_index=True)
    else:
        st.caption(
            f"Nothing picked yet — falling back to the bundled demo test set "
            f"(`{plan.test_set}`): **{load_test_set_count(plan)} questions** on the bundled "
            "demo docs (works too, but it's demo data)."
        )

    col_note, col_btn = st.columns([3, 1])
    with col_note:
        n_questions = len(custom_q) if custom_q else load_test_set_count(plan)
        st.caption(
            f"{len(selected)} experiment(s) × {n_questions} questions. "
            f"≈ {len(selected) * n_questions * 4} judge-LLM calls (Groq free tier: 1,000/day)."
        )
    with col_btn:
        start = st.button("▶ Run benchmark", type="primary", use_container_width=True)

    if start and not st.session_state.bench_running:
        bench = {
            "rows": [], "done_experiments": [], "done": False, "status": "Starting...",
            "saved": None, "error": None, "source": None,
        }
        st.session_state.bench = bench
        st.session_state.bench_running = True
        st.session_state.config = _pick_config()
        thread = threading.Thread(
            target=_run_benchmark_worker,
            args=(plan, selected, st.session_state.config.metrics, custom_q, docs_for_run),
            daemon=True,
        )
        thread.start()
        st.rerun()

    bench = st.session_state.bench
    _render_progress_block(bench, st.session_state.bench_running)


@st.fragment(run_every=2)
def _render_progress_block(bench: dict, running: bool) -> None:
    if running and not bench.get("done"):
        st.info(f"Running… {bench.get('status', '')}  \n(this view refreshes automatically every 2s — "
                "the first ~30-60s is library/model warm-up; then per-question scores appear)")
    elif bench.get("done"):
        if bench.get("error"):
            st.error(f"Benchmark failed: {bench['error']}")
        else:
            src = bench.get("source") or "test set"
            if bench.get("saved"):
                st.success(f"Benchmark done ({src}) — saved to `results/benchmark.db`.")
            else:
                st.info(f"Benchmark done ({src}) — showing results in this session (cloud files are read-only).")
    else:
        st.caption("Nothing run yet — pick experiments and press ▶ Run benchmark.")

    metric_cols = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    with st.container(border=True):
        st.markdown("**Progress — per-question scores**")
        if bench["rows"]:
            df = pd.DataFrame(bench["rows"])
            st.dataframe(
                df.style.format({c: "{:.4f}" for c in metric_cols if c in df.columns}, na_rep="-"),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("_no questions scored yet_")

    if bench.get("done") and bench["rows"] and not bench.get("error"):
        runs = _session_runs_from_bench(bench)
        if runs is not None:
            st.subheader("Aggregated scores")
            st.dataframe(
                runs.style.format(
                    {c: "{:.4f}" for c in metric_cols if c in runs.columns}, na_rep="-"
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "⬇️ Download results (CSV)",
                data=pd.DataFrame(bench["rows"]).to_csv(index=False).encode("utf-8"),
                file_name="benchmark_results.csv",
                mime="text/csv",
            )


def _render_test_set_uploader() -> list | None:
    """Optional custom test set uploader. Returns list[dict] or None."""
    st.file_uploader(
        "Upload your questions (CSV: `question,reference`  or  JSON)",
        type=["csv", "json"],
        key="bench_custom_set",
    )
    uploaded = st.session_state.get("bench_custom_set")
    if uploaded is None:
        return None
    try:
        questions = _parse_test_set_bytes(uploaded.name, uploaded.getvalue())
    except Exception as exc:
        st.warning(f"Could not parse file: {exc} — you can type your questions below instead.")
        return None
    if not questions:
        st.warning("That file had no questions (the blank template is just a header). "
                   "Fill it in or type your questions below instead.")
        return None
    return questions


def _questions_from_textareas() -> list | None:
    """Let visitors simply type their questions (and optional expected answers)."""
    st.markdown("**…or just type them here**")
    questions = st.text_area(
        "Questions — one per line",
        key="bench_q_text",
        placeholder="What is the refund policy?\nWhich product has a lifetime warranty?",
    )
    answers = st.text_area(
        "Expected answers — same order, one per line (optional)",
        key="bench_a_text",
        height=90,
    )
    q_lines = [ln.strip().strip('"') for ln in questions.splitlines() if ln.strip()]
    a_lines = [ln.strip().strip('"') for ln in answers.splitlines() if ln.strip()]
    if not q_lines:
        return None
    parsed = []
    for i, q in enumerate(q_lines):
        item = {"user_input": q}
        if i < len(a_lines) and a_lines[i]:
            item["reference"] = a_lines[i]
        parsed.append(item)
    return parsed


def _parse_test_set_bytes(filename: str, data: bytes) -> list[dict]:
    name = filename.lower()
    if name.endswith(".json"):
        payload = json.loads(data.decode("utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("questions") or payload.get("test_set") or []
        questions = payload
    else:
        try:
            df = pd.read_csv(io.BytesIO(data))
        except Exception:
            return _parse_test_set_csv_manual(data.decode("utf-8"))
        q_col = "user_input" if "user_input" in df.columns else (
            "question" if "question" in df.columns else df.columns[0]
        )
        ref_col = "reference" if "reference" in df.columns else None
        questions = []
        for _, r in df.iterrows():
            item = {"user_input": str(r[q_col])}
            if ref_col and not pd.isna(r[ref_col]) and str(r[ref_col]).strip():
                item["reference"] = str(r[ref_col])
            questions.append(item)

    parsed = []
    for q in questions:
        if isinstance(q, dict) and str(q.get("user_input", "")).strip():
            item = {"user_input": str(q["user_input"]).strip()}
            ref = q.get("reference", q.get("expected_answer", ""))
            if str(ref or "").strip():
                item["reference"] = str(ref).strip()
            parsed.append(item)
    return parsed


def _template_csv(plan, custom_q=None) -> str:
    """Template = header only by default (your document, YOUR questions).
    If a custom test set is already uploaded, pre-fill it for easy editing."""
    from evaluation.testset import load_test_set

    if custom_q is not None:
        qs = custom_q
    else:
        return "question,reference\n"  # blank: fill in your own questions
    lines = ["question,reference"]
    for q in qs:
        text = str(q["user_input"]).replace('"', '""')
        ref = str(q.get("reference", "")).replace('"', '""')
        lines.append(f'"{text}","{ref}"')
    return "\n".join(lines)


def _parse_test_set_csv_manual(text: str) -> list[dict]:
    """Forgiving CSV parse (works even on hand-made files)."""
    import csv

    rows: list[dict] = []
    reader = csv.reader(io.StringIO(text))
    header = None
    for line in reader:
        if not line or not any(cell.strip() for cell in line):
            continue
        if header is None:
            header = [c.strip().lower() for c in line]
            continue
        q = line[0] if len(line) > 0 else ""
        ref = line[1] if len(line) > 1 else ""
        rows.append({"user_input": q, "reference": ref})
    return rows


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