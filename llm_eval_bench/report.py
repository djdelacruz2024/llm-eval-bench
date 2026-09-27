"""Turns raw per-question results into the two artifacts a benchmark run
should leave behind: a CSV of every row (for further analysis) and a
markdown summary report with a per-model leaderboard.
"""

from __future__ import annotations

import csv
import sqlite3
import statistics
from pathlib import Path

from .storage import fetch_results


def export_csv(conn: sqlite3.Connection, run_id: str, out_path: str | Path) -> None:
    rows = fetch_results(conn, run_id)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 3) if values else None


def _rate(values: list[int]) -> float | None:
    values = [v for v in values if v is not None]
    return round(100 * sum(values) / len(values), 1) if values else None


def generate_markdown_report(conn: sqlite3.Connection, run_id: str) -> str:
    rows = fetch_results(conn, run_id)
    if not rows:
        return f"# Benchmark Report — {run_id}\n\nNo results found for this run.\n"

    suite = rows[0]["suite"]
    by_model: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        by_model.setdefault(row["model_name"], []).append(row)

    lines = [f"# Benchmark Report — `{run_id}` ({suite} suite)", ""]

    if suite == "golden":
        lines += [
            "| Model | Avg Accuracy (1-5) | Correct % | Avg Faithfulness | Hallucination % | Avg Empathy (1-5) | Avg Latency (s) | Total Cost ($) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for model, model_rows in sorted(by_model.items()):
            hallucination_pct = _rate(
                [1 if r["unsupported_claims"] not in (None, "[]") else 0 for r in model_rows if r["faithfulness_score"] is not None]
            )
            lines.append(
                "| {model} | {acc} | {correct}% | {faith} | {halluc}% | {emp} | {lat} | {cost} |".format(
                    model=model,
                    acc=_mean([r["accuracy_score"] for r in model_rows]),
                    correct=_rate([r["correct"] for r in model_rows]),
                    faith=_mean([r["faithfulness_score"] for r in model_rows]),
                    halluc=hallucination_pct,
                    emp=_mean([r["empathy_score"] for r in model_rows]),
                    lat=_mean([r["latency_s"] for r in model_rows]),
                    cost=round(sum(r["cost_usd"] or 0 for r in model_rows), 4),
                )
            )
    else:  # no_rag suite
        lines += [
            "| Model | Compliance % | Acknowledged Uncertainty % | Avg Latency (s) | Total Cost ($) |",
            "|---|---|---|---|---|",
        ]
        for model, model_rows in sorted(by_model.items()):
            lines.append(
                "| {model} | {compliant}% | {ack}% | {lat} | {cost} |".format(
                    model=model,
                    compliant=_rate([r["no_rag_compliant"] for r in model_rows]),
                    ack=_rate([r["acknowledged_uncertainty"] for r in model_rows]),
                    lat=_mean([r["latency_s"] for r in model_rows]),
                    cost=round(sum(r["cost_usd"] or 0 for r in model_rows), 4),
                )
            )

    lines.append("")
    return "\n".join(lines)


def write_report(conn: sqlite3.Connection, run_id: str, reports_dir: str | Path = "reports") -> Path:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    md_path = reports_dir / f"{run_id}.md"
    csv_path = reports_dir / f"{run_id}.csv"

    md_path.write_text(generate_markdown_report(conn, run_id), encoding="utf-8")
    export_csv(conn, run_id, csv_path)
    return md_path
