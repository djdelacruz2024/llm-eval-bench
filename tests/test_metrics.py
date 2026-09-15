import pytest

from llm_eval_bench.dataset import NoRagCase
from llm_eval_bench.metrics import (
    JudgeParseError,
    compute_cost_usd,
    parse_judge_json,
    score_accuracy,
    score_empathy,
    score_faithfulness,
    score_no_rag_compliance,
)


def test_score_accuracy_returns_expected_fields(judge):
    result = score_accuracy(judge, "What is X?", "X is Y.", "X is Y, based on the reference.")
    assert 1 <= result.score <= 5
    assert isinstance(result.correct, bool)
    assert result.justification


def test_score_faithfulness_returns_expected_fields(judge):
    result = score_faithfulness(judge, ["Context passage about X."], "X is described accordingly.")
    assert 0.0 <= result.faithfulness_score <= 1.0
    assert isinstance(result.unsupported_claims, list)
    assert result.has_hallucination == (len(result.unsupported_claims) > 0)


def test_score_faithfulness_requires_context(judge):
    with pytest.raises(ValueError):
        score_faithfulness(judge, [], "an answer with no context to check against")


def test_score_empathy_returns_expected_fields(judge):
    result = score_empathy(judge, "How do I manage stress?", "Here are some gentle suggestions...")
    assert 1 <= result.score <= 5
    assert result.justification


def test_score_no_rag_compliance_returns_expected_fields(judge):
    case = NoRagCase(
        id="t1",
        category="general_knowledge",
        question="What is a deductible?",
        acceptable_behaviors=["gives a correct general definition"],
        must_acknowledge_uncertainty=False,
    )
    result = score_no_rag_compliance(judge, case, "A deductible is the amount you pay before insurance kicks in.")
    assert isinstance(result.compliant, bool)
    assert isinstance(result.acknowledged_uncertainty, bool)


def test_compute_cost_usd_zero_rate_is_free():
    assert compute_cost_usd(1000, 1000, 0.0, 0.0) == 0.0


def test_compute_cost_usd_scales_with_tokens():
    cost = compute_cost_usd(prompt_tokens=1000, completion_tokens=2000,
                             cost_per_1k_prompt_tokens=0.5, cost_per_1k_completion_tokens=1.5)
    assert cost == pytest.approx(0.5 + 3.0)


def test_parse_judge_json_extracts_object_from_prose():
    raw = 'Sure, here is the result:\n{"score": 5, "justification": "great"}\nHope that helps!'
    parsed = parse_judge_json(raw)
    assert parsed == {"score": 5, "justification": "great"}


def test_parse_judge_json_raises_on_no_json():
    with pytest.raises(JudgeParseError):
        parse_judge_json("no json here at all")
