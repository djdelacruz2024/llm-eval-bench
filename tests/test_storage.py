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
