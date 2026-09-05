from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Chat with your documents",
    page_icon="📚",
    layout="wide",
)

SUPPORTED_FORMATS = ["txt", "md", "pdf"]


def _pick_config() -> object:
    import os

    from rag.config import load_config

    cfg_file = os.getenv("RAG_CONFIG", "config.yaml")
    plan = load_config(cfg_file)
    return plan.experiments[0]


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
        "your documents - powered by a local free RAG pipeline (Ollama + ChromaDB)."
    )

    if "config" not in st.session_state:
        st.session_state.config = _pick_config()
    if "tmp_dir" not in st.session_state:
        st.session_state.tmp_dir = tempfile.mkdtemp(prefix="ragdocs_")
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "history" not in st.session_state:
        st.session_state.history = []

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