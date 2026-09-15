"""Common interface every model provider implements.

Keeping this surface tiny (one method, one result type) is what lets the
benchmark harness, the judge, and the metrics code stay provider-agnostic:
swapping Ollama for Hugging Face or a hosted API never touches those layers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~0.75 words/token) used when a real tokenizer
    isn't available. Good enough for relative cost/latency comparisons
    across models in a benchmark; not meant to match a specific tokenizer.
    """
    words = len(text.split())
    return max(1, round(words / 0.75))


@dataclass
class GenerationResult:
    text: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int


class ModelProvider(ABC):
    """A single chat-capable model, addressable by name."""

    def __init__(self, name: str, model_id: str):
        self.name = name
        self.model_id = model_id

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> GenerationResult:
        """Run one prompt through the model and return text + timing/token info."""
        raise NotImplementedError


class ProviderError(RuntimeError):
    """Raised when a provider fails to produce a response (network, model, parsing)."""
