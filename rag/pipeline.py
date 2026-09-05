from __future__ import annotations

from dataclasses import dataclass, field

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from rag.config import ExperimentConfig
from rag.loader import chunk_documents
from rag.embeddings import build_embeddings

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the user's question using ONLY the "
            "provided context. If the context does not contain the answer, say you "
            "don't know instead of guessing.",
        ),
        ("human", "Context:\n-----\n{context}\n-----\n\nQuestion: {question}"),
    ]
)


@dataclass
class RetrievedResult:
    question: str
    contexts: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    answer: str = ""
    reference: str = ""
    reference_contexts: list[str] = field(default_factory=list)


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline: index -> retrieve -> generate."""

    def __init__(self, config: ExperimentConfig, documents: list[Document]) -> None:
        self.config = config
        self.embedding = build_embeddings(
            config.embedding_backend, config.embedding_model, config.provider, config.base_url
        )

        chunks = chunk_documents(documents, config.chunk_size, config.chunk_overlap)
        if not chunks:
            raise ValueError("Document corpus produced zero chunks. Check the docs_dir contents.")

        client = chromadb.EphemeralClient()
        collection_name = _safe_collection_name(config.experiment_id)
        self.store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding,
            client=client,
        )
        self.store.add_documents(chunks)

    def _generator(self) -> ChatOpenAI:
        kwargs: dict[str, object] = {"model": self.config.llm, "temperature": 0}
        if self.config.provider == "ollama":
            kwargs["base_url"] = self.config.resolved_base_url
            kwargs["api_key"] = "ollama"
        return ChatOpenAI(**kwargs)

    def answer(self, question: str, reference: str = "", reference_contexts: list[str] | None = None) -> RetrievedResult:
        docs = self.store.similarity_search(question, k=self.config.top_k)
        contexts = [d.page_content for d in docs]
        sources = [d.metadata.get("source", "?") for d in docs]

        answer = self._generate_answer(question, contexts)
        return RetrievedResult(
            question=question,
            contexts=contexts,
            sources=sources,
            answer=answer,
            reference=reference,
            reference_contexts=reference_contexts or [],
        )

    def _generate_answer(self, question: str, contexts: list[str]) -> str:
        chain = RAG_PROMPT | self._generator() | StrOutputParser()
        try:
            return chain.invoke({"context": "\n\n".join(contexts), "question": question}).strip()
        except Exception as exc:  # pragma: no cover - API failures surface at runtime
            return f"ERROR: {exc}"

    def retrieve(self, question: str) -> list[Document]:
        return self.store.similarity_search(question, k=self.config.top_k)


def _safe_collection_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
    return (cleaned or "collection")[:63]