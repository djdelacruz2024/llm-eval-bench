"""Orchestrates the main golden-dataset benchmark: for every model, answer
every question (grounded in its retrieved context), then score the answer
for accuracy, faithfulness, and empathy using the judge model. Results are
persisted to SQLite as they're produced so a long run can be inspected or
resumed from partial data.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .config import BenchmarkConfig
from .dataset import GoldenQuestion, load_golden_dataset
from .metrics import compute_cost_usd, score_accuracy, score_empathy, score_faithfulness
from .storage import ResultRow, connect, create_run, insert_result

ANSWER_SYSTEM_PROMPT = (
    "You are a helpful customer support and general-knowledge assistant. Answer "
    "clearly and accurately, in a warm and approachable tone. If context passages "
    "are provided, ground your answer in them and do not introduce claims the "
    "context does not support."
)


def _build_prompt(question: GoldenQuestion) -> str:
    if not question.context:
        return f"Question: {question.question}"
    context_block = "\n".join(f"- {c}" for c in question.context)
    return f"Context:\n{context_block}\n\nQuestion: {question.question}"


def run_golden_benchmark(
    config: BenchmarkConfig, db_path: str, run_id: str | None = None, notes: str = ""
) -> str:
    questions = load_golden_dataset(config.dataset_path)
    judge = config.judge.build()
    run_id = run_id or f"golden-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    conn = connect(db_path)
    create_run(conn, run_id, suite="golden", notes=notes)

    try:
        for model_cfg in config.models:
            provider = model_cfg.build()
            for q in questions:
                answer = provider.generate(_build_prompt(q), system=ANSWER_SYSTEM_PROMPT)

                accuracy = score_accuracy(judge, q.question, q.reference_answer, answer.text)
                empathy = score_empathy(judge, q.question, answer.text)
                faithfulness = (
                    score_faithfulness(judge, q.context, answer.text) if q.context else None
                )

                cost = compute_cost_usd(
                    answer.prompt_tokens,
                    answer.completion_tokens,
                    model_cfg.cost_per_1k_prompt_tokens,
                    model_cfg.cost_per_1k_completion_tokens,
                )

                notes_parts = [f"accuracy: {accuracy.justification}", f"empathy: {empathy.justification}"]
                if faithfulness is not None:
                    notes_parts.append(f"faithfulness: {faithfulness.justification}")

                insert_result(
                    conn,
                    ResultRow(
                        run_id=run_id,
                        suite="golden",
                        model_name=model_cfg.name,
                        question_id=q.id,
                        category=q.category,
                        question=q.question,
                        answer_text=answer.text,
                        latency_s=answer.latency_s,
                        prompt_tokens=answer.prompt_tokens,
                        completion_tokens=answer.completion_tokens,
                        cost_usd=cost,
                        accuracy_score=accuracy.score,
                        correct=accuracy.correct,
                        faithfulness_score=(faithfulness.faithfulness_score if faithfulness else None),
                        unsupported_claims=(faithfulness.unsupported_claims if faithfulness else []),
                        empathy_score=empathy.score,
                        judge_notes=" | ".join(notes_parts),
                    ),
                )
    finally:
        conn.close()

    return run_id
