"""Minimal FastAPI dashboard for browsing benchmark runs: a run list, a
per-model leaderboard with a chart, and a per-question breakdown showing each
model's response next to the ground truth it was graded against.
Reads directly from the SQLite file a benchmark run wrote to — no separate
API/database layer needed for a project this size.
"""

from __future__ import annotations

import json
import re
import sqlite3
import statistics
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..storage import connect, fetch_results, fetch_runs
from .markdown import render_markdown

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 3) if values else None


def _rate(values: list[int]) -> float | None:
    values = [v for v in values if v is not None]
    return round(100 * sum(values) / len(values), 1) if values else None


def _unsupported_claims(row: sqlite3.Row) -> list[str]:
    return json.loads(row["unsupported_claims"] or "[]")


# The golden suite joins its per-metric judge justifications into one
# judge_notes string: "accuracy: ... | empathy: ... | faithfulness: ...".
_NOTE_SEGMENT = re.compile(r"(?:^| \| )(accuracy|empathy|faithfulness): ")


def _judge_rationale(row: sqlite3.Row) -> dict[str, str]:
    notes = row["judge_notes"] or ""
    if row["suite"] != "golden":
        return {"compliance": notes} if notes else {}
    parts = _NOTE_SEGMENT.split(notes)
    # split() yields [prefix, label, text, label, text, ...]
    return {label: text.strip() for label, text in zip(parts[1::2], parts[2::2])}


def _json_or(value: str | None, default):
    return json.loads(value) if value else default


def _questions(rows: list[sqlite3.Row]) -> list[dict]:
    """Groups a run's results by question, so each question can be shown once
    with its ground truth and every model's graded response beneath it."""
    by_question: dict[str, dict] = {}
    for row in rows:
        keys = row.keys()
        question = by_question.setdefault(row["question_id"], {
            "id": row["question_id"],
            "category": row["category"],
            "question": row["question"],
            "reference_answer": row["reference_answer"] if "reference_answer" in keys else None,
            "context": _json_or(row["context"], []) if "context" in keys else [],
            "expected_behavior": (
                _json_or(row["expected_behavior"], None) if "expected_behavior" in keys else None
            ),
            "responses": [],
        })
        question["responses"].append({
            "model": row["model_name"],
            "answer_html": render_markdown(row["answer_text"]),
            "accuracy_score": row["accuracy_score"],
            "correct": row["correct"],
            "faithfulness_score": row["faithfulness_score"],
            "unsupported_claims": _unsupported_claims(row),
            "empathy_score": row["empathy_score"],
            "no_rag_compliant": row["no_rag_compliant"],
            "acknowledged_uncertainty": row["acknowledged_uncertainty"],
            "latency_s": row["latency_s"],
            "prompt_tokens": row["prompt_tokens"],
            "completion_tokens": row["completion_tokens"],
            "cost_usd": row["cost_usd"],
            "rationale": _judge_rationale(row),
        })
    return [by_question[qid] for qid in sorted(by_question)]


def _leaderboard(rows: list[sqlite3.Row], suite: str) -> list[dict]:
    by_model: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        by_model.setdefault(row["model_name"], []).append(row)

    board = []
    for model, model_rows in sorted(by_model.items()):
        entry = {
            "model": model,
            "avg_latency_s": _mean([r["latency_s"] for r in model_rows]),
            "total_cost_usd": round(sum(r["cost_usd"] or 0 for r in model_rows), 4),
        }
        if suite == "golden":
            entry.update(
                avg_accuracy=_mean([r["accuracy_score"] for r in model_rows]),
                correct_pct=_rate([r["correct"] for r in model_rows]),
                avg_faithfulness=_mean([r["faithfulness_score"] for r in model_rows]),
                # share of context-grounded answers with at least one unsupported claim
                hallucination_pct=_rate(
                    [1 if _unsupported_claims(r) else 0
                     for r in model_rows if r["faithfulness_score"] is not None]
                ),
                avg_empathy=_mean([r["empathy_score"] for r in model_rows]),
            )
        else:
            entry.update(
                compliance_pct=_rate([r["no_rag_compliant"] for r in model_rows]),
                uncertainty_ack_pct=_rate([r["acknowledged_uncertainty"] for r in model_rows]),
            )
        board.append(entry)
    return board


def create_app(db_path: str) -> FastAPI:
    app = FastAPI(title="llm-eval-bench dashboard")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    def get_conn() -> sqlite3.Connection:
        return connect(db_path)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        conn = get_conn()
        runs = fetch_runs(conn)
        conn.close()
        return templates.TemplateResponse(request, "index.html", {"runs": runs})

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_detail(request: Request, run_id: str):
        conn = get_conn()
        rows = fetch_results(conn, run_id)
        conn.close()
        suite = rows[0]["suite"] if rows else "golden"
        board = _leaderboard(rows, suite)
        return templates.TemplateResponse(
            request,
            "run_detail.html",
            {
                "run_id": run_id,
                "suite": suite,
                "board": board,
                "questions": _questions(rows),
                "models": [entry["model"] for entry in board],
            },
        )

    @app.get("/api/runs/{run_id}/summary")
    def run_summary(run_id: str):
        conn = get_conn()
        rows = fetch_results(conn, run_id)
        conn.close()
        suite = rows[0]["suite"] if rows else "golden"
        return {"run_id": run_id, "suite": suite, "leaderboard": _leaderboard(rows, suite)}

    return app
