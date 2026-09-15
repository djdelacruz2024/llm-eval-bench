"""Provider backed by a local Ollama server (https://ollama.com).

Ollama is the default backend for this project specifically so the whole
benchmark suite — including the judge model — runs on open-source models on
your own machine, with $0 API cost and no data leaving the box.
"""

from __future__ import annotations

import time

import requests

from .base import GenerationResult, ModelProvider, ProviderError


class OllamaProvider(ModelProvider):
    def __init__(self, name: str, model_id: str, base_url: str = "http://localhost:11434",
                 timeout_s: float = 120.0):
        super().__init__(name, model_id)
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def generate(self, prompt: str, system: str | None = None) -> GenerationResult:
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
        }
        start = time.perf_counter()
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate", json=payload, timeout=self.timeout_s
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise ProviderError(
                f"Ollama request failed for model '{self.model_id}' — is `ollama serve` "
                f"running and has `ollama pull {self.model_id}` been run? ({exc})"
            ) from exc

        latency_s = time.perf_counter() - start
        data = resp.json()
        return GenerationResult(
            text=data.get("response", ""),
            latency_s=latency_s,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
        )
