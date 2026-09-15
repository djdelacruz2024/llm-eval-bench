import pytest

from llm_eval_bench.providers.anthropic_provider import AnthropicProvider
from llm_eval_bench.providers.base import ProviderError


def test_anthropic_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider = AnthropicProvider(name="judge", model_id="claude-sonnet-5")

    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
        provider.generate("hello")


def test_anthropic_provider_defaults():
    provider = AnthropicProvider(name="judge")
    assert provider.model_id == "claude-sonnet-5"
    assert provider.effort == "low"
    assert provider.max_tokens == 2048


def test_config_builds_anthropic_provider(tmp_path):
    from llm_eval_bench.config import BenchmarkConfig

    config_yaml = tmp_path / "cfg.yaml"
    config_yaml.write_text(
        """
models:
  - name: mock-a
    provider: mock
judge:
  name: claude-judge
  provider: anthropic
  model_id: claude-sonnet-5
  effort: low
  max_tokens: 512
dataset_path: golden.json
no_rag_dataset_path: no_rag.json
"""
    )
    config = BenchmarkConfig.from_yaml(config_yaml)
    judge_provider = config.judge.build()

    assert isinstance(judge_provider, AnthropicProvider)
    assert judge_provider.effort == "low"
    assert judge_provider.max_tokens == 512
