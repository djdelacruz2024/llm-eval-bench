from html import escape

from fastapi.testclient import TestClient

from llm_eval_bench.benchmark import run_golden_benchmark
from llm_eval_bench.config import BenchmarkConfig
from llm_eval_bench.dashboard.app import _judge_rationale, create_app
from llm_eval_bench.dashboard.markdown import render_markdown
from llm_eval_bench.dataset import load_golden_dataset, load_no_rag_dataset
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
    assert "Hallucination %" in resp.text


def test_golden_run_detail_shows_ground_truth_and_reasoning(demo_config_path, tmp_path):
    db_path = _seed_db(demo_config_path, tmp_path)
    client = TestClient(create_app(str(db_path)))
    reference = load_golden_dataset(BenchmarkConfig.from_yaml(demo_config_path).dataset_path)[0]

    resp = client.get("/runs/dash-golden")
    assert resp.status_code == 200
    assert "Reference answer" in resp.text
    assert escape(reference.reference_answer) in resp.text
    assert escape(reference.context[0]) in resp.text
    assert "Judge's reasoning" in resp.text


def test_no_rag_run_detail_shows_behavior_rule(demo_config_path, tmp_path):
    db_path = _seed_db(demo_config_path, tmp_path)
    client = TestClient(create_app(str(db_path)))
    case = load_no_rag_dataset(BenchmarkConfig.from_yaml(demo_config_path).no_rag_dataset_path)[0]

    resp = client.get("/runs/dash-no-rag")
    assert resp.status_code == 200
    assert "Acceptable behaviors" in resp.text
    assert escape(case.acceptable_behaviors[0]) in resp.text


def test_judge_rationale_splits_golden_notes_by_metric():
    row = {
        "suite": "golden",
        "judge_notes": "accuracy: Matches. | empathy: Warm | kind. | faithfulness: All grounded.",
    }
    assert _judge_rationale(row) == {
        "accuracy": "Matches.",
        "empathy": "Warm | kind.",
        "faithfulness": "All grounded.",
    }


def test_render_markdown_formats_and_escapes():
    html = str(render_markdown("## Steps\n1. **Open** it\n2. Close <script>x</script>\n\nDone."))
    assert "<h4>Steps</h4>" in html
    assert "<ol>" in html and "<strong>Open</strong>" in html
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "<p>Done.</p>" in html


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
    assert all("hallucination_pct" in entry for entry in data["leaderboard"])
