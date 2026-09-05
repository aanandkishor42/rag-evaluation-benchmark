from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_test_set(path: Path) -> list[dict[str, Any]]:
    """Load [{question, reference, reference_contexts?}] test questions."""
    if not path.exists():
        raise FileNotFoundError(f"Test set not found: {path}")

    items = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(items, dict):
        items = items.get("questions", [])
    if not isinstance(items, list):
        raise ValueError(f"Test set must be a JSON list of questions: {path}")

    normalized = []
    for item in items:
        question = item.get("question", "").strip()
        if not question:
            continue
        normalized.append(
            {
                "user_input": question,
                "reference": item.get("reference", "").strip(),
                "reference_contexts": [c for c in item.get("reference_contexts", []) if c],
            }
        )
    return normalized