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
        effort: str | None = None,
    ):
        super().__init__(name, model_id)
        self.api_key_env = api_key_env
        self.max_tokens = max_tokens
        # `effort` is only accepted by some models (e.g. Sonnet 5, Opus 5) and
        # is rejected with a 400 on others (e.g. Haiku 4.5) — omit it unless
        # explicitly set via the config, rather than guessing per model_id.
        self.effort = effort
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        # Check the key before importing the SDK: it's configuration the user
        # always has to supply, and it keeps this check testable without the
        # optional `anthropic` package installed.
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ProviderError(
                f"Set the {self.api_key_env} environment variable with your Anthropic API "
                f"key to use model '{self.model_id}'."
            )
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderError(
                "The `anthropic` package is required for AnthropicProvider. "
                'Install it with `pip install -e ".[anthropic]"`.'
            ) from exc

        self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def generate(self, prompt: str, system: str | None = None) -> GenerationResult:
        client = self._get_client()

        kwargs = {}
        if self.effort is not None:
            kwargs["output_config"] = {"effort": self.effort}

        start = time.perf_counter()
        try:
            response = client.messages.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                system=system or "",
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
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
