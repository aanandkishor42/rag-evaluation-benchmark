from __future__ import annotations

import hashlib
import math

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from rag.config import DEFAULT_OLLAMA_BASE_URL

HASH_EMBEDDING_DIM = 256


class HashingEmbeddings(Embeddings):
    """Deterministic, offline, zero-cost embeddings for smoke tests / CI.

    Produces a fixed-size bag-of-tokens vector (token n-grams hashed into dims).
    Not semantically useful like a real embedding model, but lets the full
    chunking -> indexing -> retrieval pipeline run without any API key.
    """

    def __init__(self, model_dim: int = HASH_EMBEDDING_DIM) -> None:
        self.model_dim = model_dim

    def _vectorize(self, text: str) -> list[float]:
        vec = [0.0] * self.model_dim
        tokens = _tokenize(text)
        grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        for gram in grams:
            digest = hashlib.md5(gram.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.model_dim
            vec[index] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectorize(text)


def _tokenize(text: str) -> list[str]:
    words = []
    for token in text.lower().replace("\n", " ").split():
        cleaned = "".join(ch for ch in token if ch.isalnum())
        if cleaned:
            words.append(cleaned)
    return words


class OllamaOpenAIEmbeddings(OpenAIEmbeddings):
    """OpenAIEmbeddings pointed at Ollama, bypassing langchain's tokenizer.

    LangChain's newer embedding client tokenizes text into integer token-IDs
    before the request, which Ollama's /v1/embeddings endpoint rejects. This
    wrapper sends the raw strings (Ollama tokenizes internally).
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), self.chunk_size):
            response = self.client.create(input=texts[i : i + self.chunk_size], model=self.model)
            data = response.data if hasattr(response, "data") else response["data"]
            for item in data:
                if isinstance(item, dict):
                    results.append(item["embedding"])
                else:
                    results.append(item.embedding)
        return results

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_embeddings(
    backend: str, model: str, provider: str = "openai", base_url: str | None = None
) -> Embeddings:
    if backend == "hash":
        return HashingEmbeddings()
    if backend == "openai":
        if provider == "ollama":
            return OllamaOpenAIEmbeddings(
                model=model,
                base_url=base_url or DEFAULT_OLLAMA_BASE_URL,
                api_key="ollama",
            )
        return OpenAIEmbeddings(model=model)
    if backend in ("huggingface", "hf"):
        return _huggingface_embeddings(model)
    if backend == "fastembed":
        return _fastembed_embeddings(model)
    raise ValueError(
        f"Unknown embedding_backend '{backend}'. "
        "Use 'openai', 'hash', 'huggingface' or 'fastembed'."
    )


def _fastembed_embeddings(model: str) -> Embeddings:
    """Free embeddings that run locally on CPU (no API key, no network quota)."""
    import os

    from langchain_community.embeddings import FastEmbedEmbeddings

    model = model or "BAAI/bge-small-en-v1.5"
    return FastEmbedEmbeddings(
        model_name=model,
        max_length=512,
        cache_dir=os.path.expanduser("~/fastembed_cache"),
    )


def _huggingface_embeddings(model: str) -> Embeddings:
    """Free HuggingFace Inference API embeddings (needs HF_TOKEN in the environment)."""
    import os

    from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

    model = model or "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceInferenceAPIEmbeddings(
        model_name=model,
        api_key=os.getenv("HF_TOKEN") or "",
    )