"""Deterministic offline provider used by the test suite and for demoing the
framework's mechanics without needing Ollama, a GPU, or network access.
"""

from __future__ import annotations

import hashlib
import json
import time

from .base import GenerationResult, ModelProvider, estimate_tokens


class MockProvider(ModelProvider):
    """Returns a canned/deterministic response so unit tests don't depend on
    a real model being installed. `canned_response` lets a test pin an exact
    output; otherwise a short deterministic string is derived from the prompt.
    """

    def __init__(self, name: str, model_id: str = "mock", canned_response: str | None = None,
                 simulated_latency_s: float = 0.0):
        super().__init__(name, model_id)
        self.canned_response = canned_response
        self.simulated_latency_s = simulated_latency_s

    def generate(self, prompt: str, system: str | None = None) -> GenerationResult:
        start = time.perf_counter()
        if self.simulated_latency_s:
            time.sleep(self.simulated_latency_s)

        if self.canned_response is not None:
            text = self.canned_response
        else:
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
            text = f"[mock:{self.model_id}:{digest}] This is a deterministic stand-in response."

        latency_s = time.perf_counter() - start
        return GenerationResult(
            text=text,
            latency_s=latency_s,
            prompt_tokens=estimate_tokens(prompt) + estimate_tokens(system or ""),
            completion_tokens=estimate_tokens(text),
        )


class MockJudgeProvider(ModelProvider):
    """Stands in for the judge model in offline demos/tests. Every rubric in
    this project asks for a JSON object with a different subset of keys
    (score, correct, faithfulness_score, unsupported_claims, compliant,
    acknowledged_uncertainty, justification) — rather than parsing which
    rubric was asked, this returns a fixed superset JSON blob that satisfies
    every rubric's parser at once.
    """

    _RESPONSE = {
        "score": 4,
        "correct": True,
        "faithfulness_score": 0.9,
        "unsupported_claims": [],
        "compliant": True,
        "acknowledged_uncertainty": True,
        "justification": "Mock judge: deterministic placeholder score for offline demo/testing.",
    }

    def __init__(self, name: str, model_id: str = "mock-judge"):
        super().__init__(name, model_id)

    def generate(self, prompt: str, system: str | None = None) -> GenerationResult:
        start = time.perf_counter()
        text = json.dumps(self._RESPONSE)
        latency_s = time.perf_counter() - start
        return GenerationResult(
            text=text,
            latency_s=latency_s,
            prompt_tokens=estimate_tokens(prompt) + estimate_tokens(system or ""),
            completion_tokens=estimate_tokens(text),
        )
