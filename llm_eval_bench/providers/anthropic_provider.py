"""Provider backed by the Anthropic API (Claude) — an optional, stronger
judge than a small local model, at real per-token cost. Import of the
`anthropic` package is deferred so the rest of the package works without
it installed; install with `pip install -e ".[anthropic]"`.

Cost reference (Anthropic first-party API rates, $/1M tokens):
  Claude Sonnet 5 (claude-sonnet-5): $2 input / $10 output
  Claude Opus 5   (claude-opus-5):   $5 input / $25 output
  Claude Haiku 4.5 (claude-haiku-4-5): $1 input / $5 output
See config.claude-judge.yaml for a ready-to-use judge config with these
rates already filled into cost_per_1k_prompt_tokens / cost_per_1k_completion_tokens.
"""

from __future__ import annotations

import os
import time

from .base import GenerationResult, ModelProvider, ProviderError


class AnthropicProvider(ModelProvider):
    def __init__(
        self,
        name: str,
        model_id: str = "claude-sonnet-5",
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_tokens: int = 2048,
        effort: str = "low",
    ):
        super().__init__(name, model_id)
        self.api_key_env = api_key_env
        self.max_tokens = max_tokens
        # A judge grading a short rubric is closer to classification than
        # hard reasoning, so "low" effort keeps cost/latency down; raise it
        # via provider_kwargs in the config if judge quality looks noisy.
        self.effort = effort
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderError(
                "The `anthropic` package is required for AnthropicProvider. "
                'Install it with `pip install -e ".[anthropic]"`.'
            ) from exc

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ProviderError(
                f"Set the {self.api_key_env} environment variable with your Anthropic API "
                f"key to use model '{self.model_id}'."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def generate(self, prompt: str, system: str | None = None) -> GenerationResult:
        client = self._get_client()

        start = time.perf_counter()
        try:
            response = client.messages.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                system=system or "",
                output_config={"effort": self.effort},
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # anthropic raises a typed exception hierarchy
            raise ProviderError(
                f"Anthropic API request failed for model '{self.model_id}': {exc}"
            ) from exc
        latency_s = time.perf_counter() - start

        text = "".join(block.text for block in response.content if block.type == "text")
        return GenerationResult(
            text=text,
            latency_s=latency_s,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        )
