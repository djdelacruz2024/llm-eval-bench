from llm_eval_bench.benchmark import run_golden_benchmark
from llm_eval_bench.config import BenchmarkConfig
from llm_eval_bench.no_rag_benchmark import run_no_rag_benchmark
from llm_eval_bench.storage import connect, fetch_results


def test_run_golden_benchmark_populates_results(demo_config_path, tmp_path):
    config = BenchmarkConfig.from_yaml(demo_config_path)
    db_path = tmp_path / "results.db"

    run_id = run_golden_benchmark(config, db_path=str(db_path), run_id="test-golden")

    conn = connect(db_path)
    rows = fetch_results(conn, run_id)
    conn.close()

    num_models = len(config.models)
    num_questions = len(__import__("json").loads(open(config.dataset_path).read()))
    assert len(rows) == num_models * num_questions
    assert all(r["accuracy_score"] is not None for r in rows)
    assert all(r["empathy_score"] is not None for r in rows)


def test_run_no_rag_benchmark_populates_results(demo_config_path, tmp_path):
    config = BenchmarkConfig.from_yaml(demo_config_path)
    db_path = tmp_path / "results.db"

    run_id = run_no_rag_benchmark(config, db_path=str(db_path), run_id="test-no-rag")

    conn = connect(db_path)
    rows = fetch_results(conn, run_id)
    conn.close()

    num_models = len(config.models)
    num_cases = len(__import__("json").loads(open(config.no_rag_dataset_path).read()))
    assert len(rows) == num_models * num_cases
    assert all(r["no_rag_compliant"] is not None for r in rows)
