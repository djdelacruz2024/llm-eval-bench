import json
import sqlite3

from llm_eval_bench.storage import ResultRow, connect, create_run, fetch_results, fetch_runs, insert_result


def test_create_run_and_fetch(tmp_path):
    conn = connect(tmp_path / "test.db")
    create_run(conn, "run-1", suite="golden", notes="a test run")
    runs = fetch_runs(conn)
    assert len(runs) == 1
    assert runs[0]["run_id"] == "run-1"
    assert runs[0]["suite"] == "golden"
    conn.close()


def test_insert_and_fetch_result_round_trip(tmp_path):
    conn = connect(tmp_path / "test.db")
    create_run(conn, "run-1", suite="golden")
    insert_result(
        conn,
        ResultRow(
            run_id="run-1",
            suite="golden",
            model_name="model-a",
            question_id="g01",
            category="nutrition",
            question="How much sugar?",
            answer_text="Less than 50g/day.",
            latency_s=0.5,
            prompt_tokens=10,
            completion_tokens=5,
            cost_usd=0.0,
            accuracy_score=5,
            correct=True,
            faithfulness_score=0.95,
            unsupported_claims=["a stray claim"],
            empathy_score=4,
            judge_notes="looks right",
        ),
    )
    rows = fetch_results(conn, "run-1")
    assert len(rows) == 1
    row = rows[0]
    assert row["model_name"] == "model-a"
    assert row["accuracy_score"] == 5
    assert row["correct"] == 1
    assert row["unsupported_claims"] == '["a stray claim"]'
    conn.close()


def test_fetch_results_filters_by_run(tmp_path):
    conn = connect(tmp_path / "test.db")
    create_run(conn, "run-a", suite="golden")
    create_run(conn, "run-b", suite="golden")
    insert_result(conn, ResultRow(run_id="run-a", suite="golden", model_name="m", question_id="q1"))
    insert_result(conn, ResultRow(run_id="run-b", suite="golden", model_name="m", question_id="q1"))

    rows_a = fetch_results(conn, "run-a")
    rows_all = fetch_results(conn)

    assert len(rows_a) == 1
    assert len(rows_all) == 2
    conn.close()


def test_ground_truth_round_trips(tmp_path):
    conn = connect(tmp_path / "test.db")
    create_run(conn, "run-1", suite="golden")
    insert_result(
        conn,
        ResultRow(
            run_id="run-1", suite="golden", model_name="m", question_id="g01",
            reference_answer="100 °C", context=["passage one", "passage two"],
        ),
    )
    row = fetch_results(conn, "run-1")[0]
    assert row["reference_answer"] == "100 °C"
    assert json.loads(row["context"]) == ["passage one", "passage two"]
    assert row["expected_behavior"] is None
    conn.close()


def test_connect_adds_new_columns_to_an_older_database(tmp_path):
    db_path = tmp_path / "old.db"
    old = sqlite3.connect(db_path)
    old.executescript(
        """
        CREATE TABLE runs (run_id TEXT PRIMARY KEY, suite TEXT NOT NULL,
                           started_at TEXT NOT NULL, notes TEXT);
        CREATE TABLE results (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                              suite TEXT NOT NULL, model_name TEXT NOT NULL,
                              question_id TEXT NOT NULL, created_at TEXT NOT NULL);
        """
    )
    old.close()

    conn = connect(db_path)
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(results)")}
    assert {"reference_answer", "context", "expected_behavior"} <= columns
    conn.close()
