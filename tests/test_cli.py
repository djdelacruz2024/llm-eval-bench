from llm_eval_bench.cli import main


def test_cli_run_writes_report(demo_config_path, tmp_path):
    db_path = tmp_path / "cli.db"
    reports_dir = tmp_path / "reports"

    exit_code = main([
        "run",
        "--config", demo_config_path,
        "--db", str(db_path),
        "--run-id", "cli-golden",
        "--reports-dir", str(reports_dir),
    ])

    assert exit_code == 0
    assert (reports_dir / "cli-golden.md").exists()
    assert (reports_dir / "cli-golden.csv").exists()


def test_cli_run_no_rag_writes_report(demo_config_path, tmp_path):
    db_path = tmp_path / "cli.db"
    reports_dir = tmp_path / "reports"

    exit_code = main([
        "run-no-rag",
        "--config", demo_config_path,
        "--db", str(db_path),
        "--run-id", "cli-no-rag",
        "--reports-dir", str(reports_dir),
    ])

    assert exit_code == 0
    assert (reports_dir / "cli-no-rag.md").exists()


def test_cli_report_regenerates_existing_run(demo_config_path, tmp_path):
    db_path = tmp_path / "cli.db"
    reports_dir = tmp_path / "reports"
    main(["run", "--config", demo_config_path, "--db", str(db_path),
          "--run-id", "cli-golden", "--reports-dir", str(reports_dir)])

    (reports_dir / "cli-golden.md").unlink()
    exit_code = main(["report", "--db", str(db_path), "--run-id", "cli-golden",
                       "--reports-dir", str(reports_dir)])

    assert exit_code == 0
    assert (reports_dir / "cli-golden.md").exists()
