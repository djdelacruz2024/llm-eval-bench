"""Loads a benchmark run's YAML config: which models to evaluate, which model
acts as judge, and the $/1K-token rates used for the cost metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .providers import ModelProvider, build_provider


@dataclass
class ModelConfig:
    name: str
    provider: str
    model_id: str
    cost_per_1k_prompt_tokens: float = 0.0
    cost_per_1k_completion_tokens: float = 0.0
    provider_kwargs: dict = field(default_factory=dict)

    def build(self) -> ModelProvider:
        return build_provider(self.name, self.provider, self.model_id, **self.provider_kwargs)


@dataclass
class BenchmarkConfig:
    models: list[ModelConfig]
    judge: ModelConfig
    dataset_path: str
    no_rag_dataset_path: str

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BenchmarkConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

        def parse_model(entry: dict) -> ModelConfig:
            known = {"name", "provider", "model_id", "cost_per_1k_prompt_tokens",
                     "cost_per_1k_completion_tokens"}
            return ModelConfig(
                name=entry["name"],
                provider=entry["provider"],
                model_id=entry.get("model_id", entry["name"]),
                cost_per_1k_prompt_tokens=entry.get("cost_per_1k_prompt_tokens", 0.0),
                cost_per_1k_completion_tokens=entry.get("cost_per_1k_completion_tokens", 0.0),
                provider_kwargs={k: v for k, v in entry.items() if k not in known},
            )

        base_dir = Path(path).parent
        return cls(
            models=[parse_model(m) for m in raw["models"]],
            judge=parse_model(raw["judge"]),
            dataset_path=str(base_dir / raw.get("dataset_path", "data/golden_dataset.json")),
            no_rag_dataset_path=str(base_dir / raw.get("no_rag_dataset_path", "data/no_rag_rules.json")),
        )
