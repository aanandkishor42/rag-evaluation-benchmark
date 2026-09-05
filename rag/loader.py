from __future__ import annotations

from pathlib import Path
from typing import Sequence

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_EXTS = {".txt", ".md", ".pdf"}


def load_documents(docs_dir: Path) -> list[Document]:
    """Load all supported files in docs_dir into LangChain Documents."""
    files = [p for p in docs_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]
    if not files:
        raise FileNotFoundError(f"No supported documents (.txt/.md/.pdf) found in {docs_dir}")

    documents: list[Document] = []
    for path in sorted(files):
        if path.suffix.lower() == ".pdf":
            documents.extend(_load_pdf(path))
        else:
            try:
                loader = TextLoader(str(path), encoding="utf-8")
            except TypeError:
                loader = TextLoader(str(path))
            docs = loader.load()
            for doc in docs:
                doc.metadata = {"source": path.name}
            documents.extend(docs)
    return documents


def _load_pdf(path: Path) -> list[Document]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(
                Document(page_content=text, metadata={"source": f"{path.name} (p.{i + 1})"})
            )
    return docs


def chunk_documents(documents: Sequence[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split documents into chunks using the configured chunk size / overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    return splitter.split_documents(list(documents))  # type: ignore[arg-type]