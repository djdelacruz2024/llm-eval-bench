import json

from llm_eval_bench.benchmark import run_golden_benchmark
from llm_eval_bench.config import BenchmarkConfig
from llm_eval_bench.report import export_csv, generate_markdown_report, write_report
from llm_eval_bench.storage import connect


def _run(demo_config_path, tmp_path):
    config = BenchmarkConfig.from_yaml(demo_config_path)
    db_path = tmp_path / "results.db"
    run_id = run_golden_benchmark(config, db_path=str(db_path), run_id="report-test")
    return db_path, run_id, config


def test_generate_markdown_report_lists_every_model(demo_config_path, tmp_path):
    db_path, run_id, config = _run(demo_config_path, tmp_path)
    conn = connect(db_path)
    report = generate_markdown_report(conn, run_id)
    conn.close()

    for model in config.models:
        assert model.name in report
    assert "Avg Accuracy" in report


def test_generate_markdown_report_handles_missing_run(tmp_path):
    from llm_eval_bench.storage import connect as _connect

    conn = _connect(tmp_path / "empty.db")
    report = generate_markdown_report(conn, "does-not-exist")
    conn.close()
    assert "No results found" in report


def test_export_csv_writes_all_rows(demo_config_path, tmp_path):
    db_path, run_id, config = _run(demo_config_path, tmp_path)
    conn = connect(db_path)
    csv_path = tmp_path / "out.csv"
    export_csv(conn, run_id, csv_path)
    conn.close()

    num_questions = len(json.loads(open(config.dataset_path).read()))
    content = csv_path.read_text()
    lines = [l for l in content.splitlines() if l.strip()]
    # header + one row per (model, question)
    assert len(lines) == 1 + len(config.models) * num_questions


def test_write_report_creates_both_files(demo_config_path, tmp_path):
    db_path, run_id, _config = _run(demo_config_path, tmp_path)
    conn = connect(db_path)
    reports_dir = tmp_path / "reports"
    md_path = write_report(conn, run_id, reports_dir=reports_dir)
    conn.close()

    assert md_path.exists()
    assert (reports_dir / f"{run_id}.csv").exists()
