from __future__ import annotations

import statistics
import warnings
from typing import Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag.config import DEFAULT_OLLAMA_BASE_URL
from rag.embeddings import OllamaOpenAIEmbeddings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

from ragas import EvaluationDataset, RunConfig, evaluate  # noqa: E402
from ragas.embeddings.base import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms.base import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

METRIC_FACTORIES = {
    "context_precision": context_precision,
    "context_recall": context_recall,
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
}

VALID_METRICS = set(METRIC_FACTORIES)


def resolve_metrics(names: list[str]) -> list[str]:
    unknown = set(names) - VALID_METRICS
    if unknown:
        raise ValueError(f"Unknown metrics: {sorted(unknown)}. Valid: {sorted(VALID_METRICS)}")
    return [n for n in names if n in VALID_METRICS]


def _chat_llm(model: str, provider: str, base_url: str | None) -> ChatOpenAI:
    kwargs: dict[str, Any] = {"model": model, "temperature": 0}
    if provider in ("ollama", "groq"):
        from rag.config import api_key_for

        kwargs["base_url"] = (
            base_url
            or ("https://api.groq.com/openai/v1" if provider == "groq" else DEFAULT_OLLAMA_BASE_URL)
        )
        kwargs["api_key"] = api_key_for(provider) or "ollama"
    return ChatOpenAI(**kwargs)


def _langchain_embeddings(model: str, provider: str, base_url: str | None, embedding_backend: str):
    if embedding_backend in ("huggingface", "hf"):
        from rag.embeddings import _huggingface_embeddings

        return _huggingface_embeddings(model)
    if provider == "ollama":
        return OllamaOpenAIEmbeddings(
            model=model,
            base_url=base_url or DEFAULT_OLLAMA_BASE_URL,
            api_key="ollama",
        )
    return OpenAIEmbeddings(model=model)


def run_ragas_evaluation(
    samples: list[dict[str, Any]],
    metric_names: list[str],
    judge_llm: str,
    embedding_model: str,
    provider: str = "openai",
    base_url: str | None = None,
    embedding_backend: str = "openai",
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate generated answers + retrieved contexts with RAGAS.

    ``samples`` rows need: user_input, retrieved_contexts (list[str]),
    response, reference. Returns (aggregate scores, per-sample rows).
    """
    resolved = resolve_metrics(metric_names)

    has_reference = all(s.get("reference", "").strip() for s in samples)
    is_ollama = provider == "ollama"
    config = RunConfig(
        max_workers=2 if is_ollama else 8,
        max_retries=3,
        timeout=600 if is_ollama else 300,
    )

    llm = LangchainLLMWrapper(_chat_llm(judge_llm, provider, base_url), run_config=config)
    needs_embeddings = "answer_relevancy" in resolved
    embeddings = (
        LangchainEmbeddingsWrapper(
            _langchain_embeddings(embedding_model, provider, base_url, embedding_backend),
            run_config=config,
        )
        if needs_embeddings
        else None
    )

    metric_instances = []
    for name in resolved:
        if name == "context_recall" and not has_reference:
            continue
        metric_instances.append(METRIC_FACTORIES[name])

    if not metric_instances:
        raise ValueError("No usable metrics for this run (context_recall needs ground truth references).")

    dataset = EvaluationDataset.from_list(samples)
    result = evaluate(
        dataset=dataset,
        metrics=metric_instances,
        llm=llm,
        embeddings=embeddings,
        run_config=config,
        raise_exceptions=False,
    )

    df = result.to_pandas()
    metric_columns = [m.name for m in metric_instances]

    aggregates: dict[str, float] = {}
    for name in metric_columns:
        values = [v for v in df[name].dropna().tolist() if v is not None]
        if values:
            aggregates[name] = round(statistics.mean(values), 4)

    keep_cols = [c for c in ("user_input", "response", *metric_columns) if c in df.columns]
    per_sample = df.reset_index(drop=True)[keep_cols].to_dict(orient="records")
    return aggregates, per_sample