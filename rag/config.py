from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 80
DEFAULT_TOP_K = 3
DEFAULT_METRICS = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"


@dataclass
class ExperimentConfig:
    """Resolved settings for a single benchmark run."""

    name: str
    docs_dir: Path
    test_set: Path
    results_dir: Path
    llm: str
    judge_llm: str
    embedding_model: str
    embedding_backend: str
    provider: str = "openai"  # openai (paid cloud) | ollama (free local)
    base_url: str | None = None
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    top_k: int = DEFAULT_TOP_K
    metrics: list[str] = field(default_factory=lambda: list(DEFAULT_METRICS))

    @property
    def resolved_base_url(self) -> str:
        return self.base_url or DEFAULT_OLLAMA_BASE_URL

    @property
    def experiment_id(self) -> str:
        return f"{self.name}-{self.embedding_backend}-{self.embedding_model}"

    def to_dict(self) -> dict[str, Any]:
        out = {
            "name": self.name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "provider": self.provider,
            "llm": self.llm,
            "judge_llm": self.judge_llm,
            "embedding_backend": self.embedding_backend,
            "embedding_model": self.embedding_model,
        }
        return out


@dataclass
class BenchmarkPlan:
    """Defaults plus a list of experiments that override them."""

    docs_dir: Path
    test_set: Path
    results_dir: Path
    llm: str
    judge_llm: str
    embedding_model: str
    embedding_backend: str
    provider: str
    base_url: str | None
    metrics: list[str]
    experiments: list[ExperimentConfig]

    def resolve(self, experiment_overrides: dict[str, Any] | None = None) -> ExperimentConfig:
        defaults = {
            "docs_dir": self.docs_dir,
            "test_set": self.test_set,
            "results_dir": self.results_dir,
            "llm": self.llm,
            "judge_llm": self.judge_llm,
            "embedding_model": self.embedding_model,
            "embedding_backend": self.embedding_backend,
            "provider": self.provider,
            "base_url": self.base_url,
            "metrics": self.metrics,
        }
        merged = {**defaults, **(experiment_overrides or {})}
        if "name" not in merged:
            merged["name"] = "adhoc"
        return ExperimentConfig(**merged)


def load_config(path: str | Path) -> BenchmarkPlan:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    models = raw.get("models", {})
    retrieval = raw.get("retrieval", {})

    docs_dir = Path(raw.get("docs_dir", "data/sample_docs"))
    test_set = Path(raw.get("test_set", "data/test_set.json"))
    results_dir = Path(raw.get("results_dir", "results"))

    experiments_raw = raw.get("experiments", [])
    if not experiments_raw:
        experiments_raw = [{"name": "adhoc", **retrieval}]

    provider = models.get("provider", "openai")
    base_url = models.get("base_url")

    experiments: list[ExperimentConfig] = []
    for exp in experiments_raw:
        cfg: dict[str, Any] = copy.deepcopy(retrieval)
        cfg.update(exp)
        cfg.update(
            {
                "docs_dir": docs_dir,
                "test_set": test_set,
                "results_dir": results_dir,
                "provider": provider,
                "base_url": base_url,
                "llm": models.get("llm", "gpt-4o-mini"),
                "judge_llm": models.get("judge_llm", models.get("llm", "gpt-4o-mini")),
                "embedding_model": models.get("embeddings", "text-embedding-3-small"),
                "embedding_backend": models.get("embedding_backend", "openai"),
            }
        )
        experiments.append(ExperimentConfig(**cfg))

    return BenchmarkPlan(
        docs_dir=docs_dir,
        test_set=test_set,
        results_dir=results_dir,
        provider=provider,
        base_url=base_url,
        llm=models.get("llm", "gpt-4o-mini"),
        judge_llm=models.get("judge_llm", models.get("llm", "gpt-4o-mini")),
        embedding_model=models.get("embeddings", "text-embedding-3-small"),
        embedding_backend=models.get("embedding_backend", "openai"),
        metrics=raw.get("metrics", DEFAULT_METRICS),
        experiments=experiments,
    )