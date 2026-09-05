from __future__ import annotations

import copy
import statistics
from typing import Any

from benchmark.store import ResultsStore
from evaluation.testset import load_test_set
from rag.config import BenchmarkPlan, ExperimentConfig
from rag.loader import load_documents
from rag.pipeline import RAGPipeline

JACCARD_HIT_THRESHOLD = 0.15


def run_benchmark(
    plan: BenchmarkPlan,
    experiment_filter: list[str] | None = None,
    smoke: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Run every experiment in the plan. Returns one dict per experiment."""
    if smoke:
        return _run_smoke(plan, experiment_filter, limit)

    test_questions = load_test_set(plan.test_set)
    if limit:
        test_questions = test_questions[:limit]

    documents = load_documents(plan.docs_dir)
    experiments = _select_experiments(plan, experiment_filter)
    store = ResultsStore(plan.results_dir / "benchmark.db")

    runs: list[dict[str, Any]] = []
    for experiment in experiments:
        pipeline = RAGPipeline(experiment, documents)
        samples = []
        for q in test_questions:
            result = pipeline.answer(q["user_input"], q.get("reference", ""), q.get("reference_contexts"))
            samples.append(
                {
                    "user_input": q["user_input"],
                    "retrieved_contexts": result.contexts,
                    "response": result.answer,
                    "reference": result.reference,
                }
            )

        print(f"\n[{experiment.name}] evaluating {len(samples)} questions with "
              f"{', '.join(experiment.metrics)} ...")
        aggregates, per_sample = evaluate_samples(experiment, samples)
        store.insert(experiment.name, experiment.to_dict(), aggregates)
        runs.append(
            {
                "experiment": experiment.name,
                "config": experiment.to_dict(),
                "scores": aggregates,
                "per_question": per_sample,
            }
        )

    store.export_csv(plan.results_dir / "results.csv", _flatten_runs(runs))
    _export_per_question(runs, plan.results_dir / "per_question.csv")
    store.close()
    return runs


def evaluate_samples(
    experiment: ExperimentConfig, samples: list[dict[str, Any]]
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    from evaluation.ragas_eval import run_ragas_evaluation

    return run_ragas_evaluation(
        samples=samples,
        metric_names=experiment.metrics,
        judge_llm=experiment.judge_llm,
        embedding_model=experiment.embedding_model,
        provider=experiment.provider,
        base_url=experiment.base_url,
        embedding_backend=experiment.embedding_backend,
    )


def _run_smoke(
    plan: BenchmarkPlan, experiment_filter: list[str] | None, limit: int | None
) -> list[dict[str, Any]]:
    test_questions = load_test_set(plan.test_set)
    if limit:
        test_questions = test_questions[:limit]

    documents = load_documents(plan.docs_dir)
    experiments = _select_experiments(plan, experiment_filter)
    if not experiments:
        experiments = [
            ExperimentConfig(
                name="smoke",
                docs_dir=plan.docs_dir,
                test_set=plan.test_set,
                results_dir=plan.results_dir,
                llm=plan.llm,
                judge_llm=plan.judge_llm,
                embedding_model=plan.embedding_model,
                embedding_backend="hash",
            )
        ]

    runs: list[dict[str, Any]] = []
    for experiment in experiments:
        smoke_config = _smoke_config(experiment)
        pipeline = RAGPipeline(smoke_config, documents)
        hits, jaccards, retrieved_counts = [], [], []
        for q in test_questions:
            docs = pipeline.retrieve(q["user_input"])
            contexts = [d.page_content for d in docs]
            reference = q.get("reference", "")
            best = _best_jaccard(contexts, reference)
            hits.append(1 if best >= JACCARD_HIT_THRESHOLD else 0)
            jaccards.append(best)
            retrieved_counts.append(len(contexts))

        scores = {
            "reference_hit_rate": round(statistics.mean(hits), 4),
            "avg_jaccard": round(statistics.mean(jaccards), 4),
            "avg_retrieved": round(statistics.mean(retrieved_counts), 2),
        }
        runs.append(
            {
                "experiment": smoke_config.name,
                "config": smoke_config.to_dict(),
                "scores": scores,
                "per_question": [
                    {"user_input": q["user_input"], "reference": q.get("reference", "")}
                    for q in test_questions
                ],
            }
        )
        print(f"{smoke_config.name}: chunk_size={smoke_config.chunk_size}, "
              f"overlap={smoke_config.chunk_overlap}, top_k={smoke_config.top_k} -> {scores}")

    _export_smoke(runs, plan.results_dir / "smoke_retrieval.csv")
    return runs


def _smoke_config(experiment: ExperimentConfig) -> ExperimentConfig:
    cfg = copy.deepcopy(experiment)
    cfg.embedding_backend = "hash"
    cfg.embedding_model = "local-hash-256"
    return cfg


def _best_jaccard(contexts: list[str], reference: str) -> float:
    if not reference or not contexts:
        return 0.0
    ref_tokens = set(_tokenize(reference))
    if not ref_tokens:
        return 0.0
    best = 0.0
    for context in contexts:
        ctx_tokens = set(_tokenize(context))
        if not ctx_tokens:
            continue
        intersection = len(ref_tokens & ctx_tokens)
        union = len(ref_tokens | ctx_tokens)
        best = max(best, intersection / union)
    return best


def _tokenize(text: str) -> list[str]:
    words = []
    for token in text.lower().replace("\n", " ").split():
        cleaned = "".join(ch for ch in token if ch.isalnum())
        if cleaned:
            words.append(cleaned)
    return words


def _select_experiments(plan: BenchmarkPlan, experiment_filter: list[str] | None) -> list[ExperimentConfig]:
    if not experiment_filter:
        return plan.experiments
    selected = [e for e in plan.experiments if e.name in experiment_filter]
    if not selected:
        raise ValueError(f"No experiments match filter {experiment_filter}. Available: "
                         f"{[e.name for e in plan.experiments]}")
    return selected


def _flatten_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        row = {"experiment": run["experiment"], "config": run["config"]}
        row.update(run["scores"])
        rows.append(row)
    return rows


def _export_per_question(runs: list[dict[str, Any]], path) -> None:
    import csv

    rows: list[dict[str, Any]] = []
    for run in runs:
        for pq in run["per_question"]:
            row = {
                "experiment": run["experiment"],
                "question": pq.get("user_input", ""),
                "response": pq.get("response", ""),
            }
            for col in pq:
                if col not in ("user_input", "response"):
                    row[col] = pq[col]
            rows.append(row)
    if rows:
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def _export_smoke(runs: list[dict[str, Any]], path) -> None:
    import csv

    rows = [
        {
            "experiment": run["experiment"],
            "chunk_size": run["config"]["chunk_size"],
            "chunk_overlap": run["config"]["chunk_overlap"],
            "top_k": run["config"]["top_k"],
            **run["scores"],
        }
        for run in runs
    ]
    if rows:
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)