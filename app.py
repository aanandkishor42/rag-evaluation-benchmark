from __future__ import annotations

import io
import json
import os
import tempfile
import time
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


def _active_config_name() -> str:
    return os.getenv("RAG_CONFIG", "config.yaml")


def _load_plan() -> object:
    from rag.config import load_config

    return load_config(os.getenv("RAG_CONFIG", "config.yaml"))


def _ping_judge(exp) -> str:
    """Cheap 1-call ping to the judge provider so failures show *before* the run."""
    try:
        from evaluation.ragas_eval import _chat_llm

        llm = _chat_llm(exp.judge_llm, exp.provider, exp.base_url)
        out = llm.invoke("Reply with exactly: OK")
        text = (getattr(out, "content", "") or "").strip()
        return "Judge online ✓ (provider responded)" if text else "Judge online ✓"
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).replace("\n", " ")[:300]
        return f"⚠️ Judge check FAILED — {msg}"


def _run_benchmark_sync(
    plan, exp_names: list[str], metrics: list[str], custom_questions=None, documents=None
) -> dict:
    """Runs the RAGAS benchmark synchronously inside an st.status() block so every
    per-question row streams LIVE to the page. Returns the bench dict."""
    from evaluation.ragas_eval import run_ragas_evaluation
    from evaluation.testset import load_test_set
    from rag.loader import load_documents
    from rag.pipeline import RAGPipeline

    bench = {"rows": [], "done_experiments": [], "done": False,
             "status": "", "saved": None, "error": None, "source": None, "ping": None}
    metric_cols = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    with st.status(
        "Running benchmark… (don't refresh this page)", expanded=True
    ) as status:
        st.write("Warming up libraries + embedding model… (first time ~30-60s)")

        documents = documents if documents is not None else load_documents(plan.docs_dir)
        test_questions = custom_questions or load_test_set(plan.test_set)
        experiments = [e for e in plan.experiments if e.name in exp_names]
        first_exp = experiments[0]
        st.write(
            f"Judge: **`{first_exp.provider}`** • model `{first_exp.judge_llm}` — "
            f"embeddings: `{first_exp.embedding_backend}`/`{first_exp.embedding_model}`"
        )
        bench["provider"] = f"{first_exp.provider} · {first_exp.judge_llm}"
        ping = _ping_judge(first_exp)
        bench["ping"] = ping
        st.write(ping)
        total = len(experiments) * len(test_questions)
        done = 0
        progress = st.progress(0.0)
        table = st.empty()

        for exp in experiments:
            status.update(label=f"Indexing documents with `{exp.name}`…")
            pipeline = RAGPipeline(exp, documents)
            for qi, q in enumerate(test_questions, 1):
                q_label = q["user_input"][:50]
                t0 = time.time()
                status.update(label=f"💬 Answering ({qi}/{len(test_questions)}): {q_label}…")
                result = pipeline.answer(q["user_input"], q.get("reference", ""), q.get("reference_contexts"))
                status.update(label=f"🔍 Scoring ({qi}/{len(test_questions)}): {q_label}… (takes ~1-3 min on local Ollama)")
                sample = [
                    {
                        "user_input": q["user_input"],
                        "retrieved_contexts": result.contexts,
                        "response": result.answer,
                        "reference": result.reference,
                    }
                ]
                _, per = run_ragas_evaluation(
                    sample, metrics, exp.judge_llm, exp.embedding_model,
                    exp.provider, exp.base_url, exp.embedding_backend,
                )
                row: dict[str, object] = {
                    "experiment": exp.name,
                    "question": q["user_input"],
                    "answer": result.answer,
                }
                if per:
                    for k, v in per[0].items():
                        if k in ("user_input", "response"):
                            continue
                        if isinstance(v, float) and pd.isna(v):
                            continue
                        row[k] = round(v, 4) if isinstance(v, float) else v
                scored = any(k in row for k in metric_cols)
                if not scored:
                    row["status"] = (
                        "no answer (refusal/error)" if not (result.answer or "").strip()
                        else "judge didn't score (quota / connection?)"
                    )
                else:
                    row["status"] = "ok"
                bench["rows"].append(row)
                done += 1
                elapsed = int(time.time() - t0)
                df = pd.DataFrame(bench["rows"])
                table.dataframe(
                    df.style.format({c: "{:.4f}" for c in metric_cols if c in df.columns},
                                    na_rep="-"),
                    use_container_width=True, hide_index=True,
                )
                progress.progress(done / total)
                status.update(label=f"✅ {done}/{total} done ({elapsed}s so far) — next question…")

        unscored = [r for r in bench["rows"] if r.get("status") != "ok"]
        label = f"Benchmark finished — {len(unscored)}/{len(bench['rows'])} question(s) couldn't be scored."
        label += " Check the `status` column & judge ping above." if unscored else " All questions scored ✓"
        status.update(label=label, state="complete")

    bench["done_experiments"] = exp_names
    bench["source"] = "custom" if custom_questions else "test_set.json"
    bench["done"] = True

    try:
        from benchmark.store import ResultsStore

        _save_run_to_disk(plan, experiments, metrics, bench)
        bench["saved"] = True
    except Exception:
        bench["saved"] = False  # read-only cloud mount -> keep results in-memory only

    st.session_state.bench = bench
    return bench


def _save_run_to_disk(plan, experiments, metrics: list[str], bench: dict) -> None:
    from benchmark.store import ResultsStore

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
    if "bench" not in st.session_state:
        st.session_state.bench = {
            "rows": [], "done_experiments": [], "done": False, "status": "",
            "saved": None, "error": None, "source": None,
        }

    cfg = st.session_state.config
    st.sidebar.markdown(
        f"**Active config:** `{_active_config_name()}`  \n"
        f"Judge: **{cfg.provider}** • `{cfg.judge_llm}`  \n"
        f"Embeddings: `{cfg.embedding_backend}` / `{cfg.embedding_model}`"
    )

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

    using_own_docs = docs_choice == "Upload my own documents" and bool(docs_for_run)
    questions_mismatch = using_own_docs and not custom_q

    if custom_q:
        preview = pd.DataFrame(
            {
                "question": [q["user_input"] for q in custom_q],
                "expected answer (reference)": [q.get("reference", "") for q in custom_q],
            }
        )
        st.success(f"Using your {q_source}: {len(custom_q)} questions.")
        st.dataframe(preview, use_container_width=True, hide_index=True)
    elif questions_mismatch:
        st.error(
            "⚠️ You uploaded your own document, but haven't given any questions yet. "
            "The bundled demo questions are about a **different** sample document and "
            "will NOT match your file — running now would give meaningless results. "
            "Please type or upload questions about YOUR document above before running."
        )
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
        start = st.button(
            "▶ Run benchmark", type="primary", use_container_width=True,
            disabled=questions_mismatch,
        )

    if start:
        _run_benchmark_sync(plan, selected, st.session_state.config.metrics, custom_q, docs_for_run)
        st.rerun()

    bench = st.session_state.bench
    _render_bench_results(bench)


@st.fragment
def _render_bench_results(bench: dict) -> None:
    metric_cols = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]

    if not bench.get("done"):
        if bench["rows"]:
            st.warning("Previous run was interrupted — showing partial results below.")
        else:
            st.info("Nothing run yet — pick your document, questions and experiment, "
                    "then press ▶ Run benchmark.")
            return

    if bench.get("error"):
        st.error(f"Benchmark failed: {bench['error']}")

    ping = bench.get("ping") or ""
    if ping.startswith("⚠️"):
        st.error(
            f"❌ **Judge: {bench.get('provider', '?')} — check FAILED before the run**, "
            "that's why scores are empty. Most likely the free tier is out of quota. "
            "Check <a href='https://console.groq.com/usage' target='_blank'>console.groq.com/usage</a> "
            "or wait for the daily reset.",
            unsafe_allow_html=True,
        )
    elif bench.get("provider"):
        st.caption(f"Run judge: {bench['provider']}")

    with st.container(border=True):
        st.markdown(
            "**Per-question scores** — each value is 0-1: "
            "`context_precision` (relevant chunks shown), `context_recall` (right chunk found), "
            "`faithfulness` (no hallucination — 🎯 the match-with-your-doc check), "
            "`answer_relevancy` (answers the question). Empty = doesn't apply to that question."
        )
        if bench["rows"]:
            df = pd.DataFrame(bench["rows"])
            if "answer" in df.columns:
                df["answer"] = df["answer"].fillna("").astype(str).str.slice(0, 160)
            st.dataframe(
                df.style.format({c: "{:.4f}" for c in metric_cols if c in df.columns}, na_rep="-"),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("_no questions scored yet_")

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
        if bench.get("saved") is False:
            st.info("Results kept in this session only (cloud files are read-only). "
                    "Run locally with Ollama to persist them.")
        elif bench.get("saved") is True:
            st.success("Saved to `results/benchmark.db`.")


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