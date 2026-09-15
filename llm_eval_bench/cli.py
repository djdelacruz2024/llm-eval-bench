"""Command-line entry point.

    llm-eval-bench run          --config config.yaml [--run-id ID] [--notes "..."]
    llm-eval-bench run-no-rag   --config config.yaml [--run-id ID] [--notes "..."]
    llm-eval-bench report       --db results.db --run-id ID [--reports-dir reports/]
    llm-eval-bench serve        --db results.db [--host 0.0.0.0] [--port 8000]
"""

from __future__ import annotations

import argparse
import sys

from .benchmark import run_golden_benchmark
from .config import BenchmarkConfig
from .no_rag_benchmark import run_no_rag_benchmark
from .report import generate_markdown_report, write_report
from .storage import connect


def _add_common_run_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--config", required=True, help="Path to benchmark YAML config")
    sub.add_argument("--db", default="results.db", help="SQLite results database path")
    sub.add_argument("--run-id", default=None, help="Custom run ID (default: timestamp-based)")
    sub.add_argument("--notes", default="", help="Freeform notes stored with the run")
    sub.add_argument("--reports-dir", default="reports", help="Directory to write the report into")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-eval-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the golden-dataset benchmark")
    _add_common_run_args(run_parser)

    no_rag_parser = subparsers.add_parser("run-no-rag", help="Run the no-RAG robustness benchmark")
    _add_common_run_args(no_rag_parser)

    report_parser = subparsers.add_parser("report", help="Regenerate a report for an existing run")
    report_parser.add_argument("--db", default="results.db")
    report_parser.add_argument("--run-id", required=True)
    report_parser.add_argument("--reports-dir", default="reports")

    serve_parser = subparsers.add_parser("serve", help="Launch the results dashboard")
    serve_parser.add_argument("--db", default="results.db")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)

    if args.command in ("run", "run-no-rag"):
        config = BenchmarkConfig.from_yaml(args.config)
        run_fn = run_golden_benchmark if args.command == "run" else run_no_rag_benchmark
        run_id = run_fn(config, db_path=args.db, run_id=args.run_id, notes=args.notes)

        conn = connect(args.db)
        md_path = write_report(conn, run_id, reports_dir=args.reports_dir)
        conn.close()

        print(f"Run complete: {run_id}")
        print(f"Report written to: {md_path}")
        return 0

    if args.command == "report":
        conn = connect(args.db)
        md_path = write_report(conn, args.run_id, reports_dir=args.reports_dir)
        conn.close()
        print(f"Report written to: {md_path}")
        return 0

    if args.command == "serve":
        import uvicorn

        from .dashboard.app import create_app

        uvicorn.run(create_app(args.db), host=args.host, port=args.port)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
