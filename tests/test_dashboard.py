from fastapi.testclient import TestClient

from llm_eval_bench.benchmark import run_golden_benchmark
from llm_eval_bench.config import BenchmarkConfig
from llm_eval_bench.dashboard.app import create_app
from llm_eval_bench.no_rag_benchmark import run_no_rag_benchmark


def _seed_db(demo_config_path, tmp_path):
    config = BenchmarkConfig.from_yaml(demo_config_path)
    db_path = tmp_path / "dashboard.db"
    run_golden_benchmark(config, db_path=str(db_path), run_id="dash-golden")
    run_no_rag_benchmark(config, db_path=str(db_path), run_id="dash-no-rag")
    return db_path


def test_index_lists_runs(demo_config_path, tmp_path):
    db_path = _seed_db(demo_config_path, tmp_path)
    client = TestClient(create_app(str(db_path)))

    resp = client.get("/")
    assert resp.status_code == 200
    assert "dash-golden" in resp.text
    assert "dash-no-rag" in resp.text


def test_golden_run_detail_renders_leaderboard(demo_config_path, tmp_path):
    db_path = _seed_db(demo_config_path, tmp_path)
    client = TestClient(create_app(str(db_path)))

    resp = client.get("/runs/dash-golden")
    assert resp.status_code == 200
    assert "Avg Accuracy" in resp.text


def test_no_rag_run_detail_renders_leaderboard(demo_config_path, tmp_path):
    db_path = _seed_db(demo_config_path, tmp_path)
    client = TestClient(create_app(str(db_path)))

    resp = client.get("/runs/dash-no-rag")
    assert resp.status_code == 200
    assert "Compliance" in resp.text


def test_api_summary_returns_leaderboard_json(demo_config_path, tmp_path):
    db_path = _seed_db(demo_config_path, tmp_path)
    client = TestClient(create_app(str(db_path)))

    resp = client.get("/api/runs/dash-golden/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == "dash-golden"
    assert len(data["leaderboard"]) > 0
