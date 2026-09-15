import json

from llm_eval_bench.providers.mock_provider import MockJudgeProvider, MockProvider


def test_mock_provider_is_deterministic_for_same_prompt():
    provider = MockProvider(name="a", model_id="mock")
    r1 = provider.generate("What is the sky made of?")
    r2 = provider.generate("What is the sky made of?")
    assert r1.text == r2.text


def test_mock_provider_differs_for_different_prompts():
    provider = MockProvider(name="a", model_id="mock")
    r1 = provider.generate("prompt one")
    r2 = provider.generate("prompt two")
    assert r1.text != r2.text


def test_mock_provider_canned_response_is_used():
    provider = MockProvider(name="a", canned_response="fixed answer")
    result = provider.generate("anything")
    assert result.text == "fixed answer"


def test_mock_provider_token_counts_are_positive():
    provider = MockProvider(name="a")
    result = provider.generate("a reasonably long prompt with several words in it")
    assert result.prompt_tokens > 0
    assert result.completion_tokens > 0


def test_mock_judge_provider_returns_parseable_json():
    judge = MockJudgeProvider(name="judge")
    result = judge.generate("Respond with ONLY this JSON object: {\"score\": <1-5>}")
    parsed = json.loads(result.text)
    assert "score" in parsed
    assert "justification" in parsed
