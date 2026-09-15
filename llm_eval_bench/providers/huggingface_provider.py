"""Provider backed by a local Hugging Face `transformers` text-generation
pipeline. Heavier to start up than Ollama but useful for models you want to
load directly (e.g. for logit-level experiments) rather than through a
server. Import of `transformers` is deferred so the rest of the package
works without it installed.
"""

from __future__ import annotations

import time

from .base import GenerationResult, ModelProvider, ProviderError, estimate_tokens


class HuggingFaceProvider(ModelProvider):
    def __init__(self, name: str, model_id: str, max_new_tokens: int = 512, device: str = "cpu"):
        super().__init__(name, model_id)
        self.max_new_tokens = max_new_tokens
        self.device = device
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ProviderError(
                "The `transformers` package is required for HuggingFaceProvider. "
                "Install it with `pip install transformers torch`."
            ) from exc

        self._pipeline = pipeline(
            "text-generation", model=self.model_id, device=self.device
        )
        return self._pipeline

    def generate(self, prompt: str, system: str | None = None) -> GenerationResult:
        pipe = self._load()
        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        start = time.perf_counter()
        try:
            output = pipe(
                full_prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_full_text=False,
            )
        except Exception as exc:  # pragma: no cover - depends on local model/hardware
            raise ProviderError(f"Hugging Face generation failed for '{self.model_id}': {exc}") from exc
        latency_s = time.perf_counter() - start

        text = output[0]["generated_text"] if output else ""
        return GenerationResult(
            text=text,
            latency_s=latency_s,
            prompt_tokens=estimate_tokens(full_prompt),
            completion_tokens=estimate_tokens(text),
        )
