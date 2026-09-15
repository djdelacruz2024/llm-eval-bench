from .base import GenerationResult, ModelProvider, ProviderError, estimate_tokens
from .huggingface_provider import HuggingFaceProvider
from .mock_provider import MockJudgeProvider, MockProvider
from .ollama_provider import OllamaProvider

_PROVIDER_CLASSES = {
    "mock": MockProvider,
    "mock_judge": MockJudgeProvider,
    "ollama": OllamaProvider,
    "huggingface": HuggingFaceProvider,
}


def build_provider(name: str, provider: str, model_id: str, **kwargs) -> ModelProvider:
    """Instantiate a provider by its config-file key (e.g. "ollama")."""
    try:
        cls = _PROVIDER_CLASSES[provider]
    except KeyError as exc:
        raise ValueError(
            f"Unknown provider '{provider}'. Available: {sorted(_PROVIDER_CLASSES)}"
        ) from exc
    return cls(name=name, model_id=model_id, **kwargs)


__all__ = [
    "GenerationResult",
    "ModelProvider",
    "ProviderError",
    "estimate_tokens",
    "HuggingFaceProvider",
    "MockProvider",
    "MockJudgeProvider",
    "OllamaProvider",
    "build_provider",
]
