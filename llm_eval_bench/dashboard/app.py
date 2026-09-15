"""Minimal FastAPI dashboard for browsing benchmark runs: a run list, a
per-model leaderboard with a chart, and a per-question drill-down table.
Reads directly from the SQLite file a benchmark run wrote to — no separate
API/database layer needed for a project this size.
"""

from __future__ import annotations

import sqlite3
import statistics
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..storage import connect, fetch_results, fetch_runs

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 3) if values else None


def _rate(values: list[int]) -> float | None:
    values = [v for v in values if v is not None]
    return round(100 * sum(values) / len(values), 1) if values else None


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
            {"run_id": run_id, "suite": suite, "board": board, "rows": rows},
        )

    @app.get("/api/runs/{run_id}/summary")
    def run_summary(run_id: str):
        conn = get_conn()
        rows = fetch_results(conn, run_id)
        conn.close()
        suite = rows[0]["suite"] if rows else "golden"
        return {"run_id": run_id, "suite": suite, "leaderboard": _leaderboard(rows, suite)}

    return app
