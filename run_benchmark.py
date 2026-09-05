from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_benchmark",
        description=(
            "RAG Evaluation & Benchmarking Pipeline - compare chunk size, overlap, "
            "top-k and embedding configs using RAGAS metrics."
        ),
    )
    parser.add_argument("--config", default="config.yaml", help="Path to benchmark config YAML")
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Only run these experiment names (from config.yaml)",
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=None,
        help="Limit to the first N test questions (useful for a quick run)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Offline self-check: no API key needed. Uses local hash embeddings, "
             "verifies chunking/indexing/retrieval, reports retrieval hit rate.",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Print previously recorded runs from the results database, then exit",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate results/report.html (charts + tables) from the results database, then exit",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    load_dotenv()

    from rag.config import load_config

    if args.smoke:
        from benchmark.runner import run_benchmark
        from benchmark.report import print_comparison

        plan = load_config(args.config)
        runs = run_benchmark(plan, experiment_filter=args.experiments, smoke=True, limit=args.questions)
        print_comparison(runs)
        print("Smoke run complete. No API calls were made. Now add your OPENAI_API_KEY "
              "and run without --smoke for the full RAGAS evaluation.")
        return 0

    if args.history:
        from benchmark.store import ResultsStore
        from benchmark.report import print_history

        plan = load_config(args.config)
        store = ResultsStore(plan.results_dir / "benchmark.db")
        print_history(store.history())
        store.close()
        return 0

    if args.report:
        from benchmark.store import ResultsStore
        from benchmark.report import write_html_report

        plan = load_config(args.config)
        store = ResultsStore(plan.results_dir / "benchmark.db")
        path = write_html_report(store.history(), plan.results_dir / "report.html")
        store.close()
        print(f"Report written to: {path}")
        print("Open it in your browser by double-clicking the file, or run:  start report.html")
        return 0

    plan = load_config(args.config)

    from benchmark.runner import run_benchmark
    from benchmark.report import print_comparison

    runs = run_benchmark(plan, experiment_filter=args.experiments, smoke=False, limit=args.questions)
    print_comparison(runs)
    print(f"Saved: {plan.results_dir / 'results.csv'} and {plan.results_dir / 'per_question.csv'}")
    print("Use --history to review past runs and catch regressions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())