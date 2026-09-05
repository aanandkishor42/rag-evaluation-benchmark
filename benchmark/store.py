from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

METRIC_COLUMNS = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]


class ResultsStore:
    """Persists benchmark runs to SQLite and exports a CSV summary."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                experiment TEXT,
                config_json TEXT,
                context_precision REAL,
                context_recall REAL,
                faithfulness REAL,
                answer_relevancy REAL
            )
            """
        )
        self._conn.commit()

    def insert(self, experiment_name: str, config: dict[str, Any], scores: dict[str, float]) -> None:
        row = (
            datetime.now(timezone.utc).isoformat(),
            experiment_name,
            json.dumps(config, sort_keys=True),
            *[scores.get(col) for col in METRIC_COLUMNS],
        )
        self._conn.execute(
            "INSERT INTO runs (created_at, experiment, config_json, context_precision, context_recall, faithfulness, answer_relevancy) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            row,
        )
        self._conn.commit()

    def history(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, created_at, experiment, context_precision, context_recall, faithfulness, answer_relevancy "
            "FROM runs ORDER BY id DESC LIMIT 100"
        ).fetchall()
        return [
            {
                "id": r[0],
                "created_at": r[1],
                "experiment": r[2],
                "context_precision": r[3],
                "context_recall": r[4],
                "faithfulness": r[5],
                "answer_relevancy": r[6],
            }
            for r in rows
        ]

    def export_csv(self, csv_path: Path, rows: list[dict[str, Any]]) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["experiment"])
            writer.writeheader()
            writer.writerows(rows)

    def close(self) -> None:
        self._conn.close()