"""SQLite persistence for benchmark runs. One `results` row per
(run, model, question). Kept deliberately flat/denormalized so the
dashboard and report layer can query it with plain SQL and no joins.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    suite TEXT NOT NULL,              -- 'golden' or 'no_rag'
    started_at TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    suite TEXT NOT NULL,
    model_name TEXT NOT NULL,
    question_id TEXT NOT NULL,
    category TEXT,
    question TEXT,
    reference_answer TEXT,            -- golden suite: the ground-truth answer
    context TEXT,                     -- golden suite: JSON list of retrieved passages
    expected_behavior TEXT,           -- no-RAG suite: JSON behavior rule
    answer_text TEXT,
    latency_s REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cost_usd REAL,
    accuracy_score REAL,
    correct INTEGER,
    faithfulness_score REAL,
    unsupported_claims TEXT,
    empathy_score REAL,
    no_rag_compliant INTEGER,
    acknowledged_uncertainty INTEGER,
    judge_notes TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_results_model ON results(model_name);
"""

# Columns added after the first release. `connect` adds any that are missing
# so databases written by older versions still open.
_ADDED_RESULT_COLUMNS = {
    "reference_answer": "TEXT",
    "context": "TEXT",
    "expected_behavior": "TEXT",
}


@dataclass
class ResultRow:
    run_id: str
    suite: str
    model_name: str
    question_id: str
    category: str | None = None
    question: str | None = None
    reference_answer: str | None = None
    context: list[str] = field(default_factory=list)
    expected_behavior: dict | None = None
    answer_text: str | None = None
    latency_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    accuracy_score: float | None = None
    correct: bool | None = None
    faithfulness_score: float | None = None
    unsupported_claims: list[str] = field(default_factory=list)
    empathy_score: float | None = None
    no_rag_compliant: bool | None = None
    acknowledged_uncertainty: bool | None = None
    judge_notes: str = ""


def connect(db_path: str | Path) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(results)")}
    for column, col_type in _ADDED_RESULT_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE results ADD COLUMN {column} {col_type}")
    conn.commit()
    return conn


def create_run(conn: sqlite3.Connection, run_id: str, suite: str, notes: str = "") -> None:
    conn.execute(
        "INSERT INTO runs (run_id, suite, started_at, notes) VALUES (?, ?, ?, ?)",
        (run_id, suite, datetime.now(timezone.utc).isoformat(), notes),
    )
    conn.commit()


def insert_result(conn: sqlite3.Connection, row: ResultRow) -> None:
    conn.execute(
        """
        INSERT INTO results (
            run_id, suite, model_name, question_id, category, question,
            reference_answer, context, expected_behavior, answer_text,
            latency_s, prompt_tokens, completion_tokens, cost_usd,
            accuracy_score, correct, faithfulness_score, unsupported_claims,
            empathy_score, no_rag_compliant, acknowledged_uncertainty, judge_notes, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row.run_id, row.suite, row.model_name, row.question_id, row.category, row.question,
            row.reference_answer, json.dumps(row.context),
            json.dumps(row.expected_behavior) if row.expected_behavior is not None else None,
            row.answer_text, row.latency_s, row.prompt_tokens, row.completion_tokens, row.cost_usd,
            row.accuracy_score, _bool_to_int(row.correct), row.faithfulness_score,
            json.dumps(row.unsupported_claims), row.empathy_score,
            _bool_to_int(row.no_rag_compliant), _bool_to_int(row.acknowledged_uncertainty),
            row.judge_notes, datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def fetch_results(conn: sqlite3.Connection, run_id: str | None = None) -> list[sqlite3.Row]:
    if run_id:
        return conn.execute(
            "SELECT * FROM results WHERE run_id = ? ORDER BY model_name, question_id", (run_id,)
        ).fetchall()
    return conn.execute("SELECT * FROM results ORDER BY run_id, model_name, question_id").fetchall()


def fetch_runs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM runs ORDER BY started_at DESC").fetchall()


def _bool_to_int(value: bool | None) -> int | None:
    return None if value is None else int(bool(value))
