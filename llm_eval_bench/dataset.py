"""Loaders for the two benchmark datasets.

`golden_dataset.json` — question/reference-answer pairs, each with the
context passages a RAG pipeline would have retrieved. Used for the main
accuracy/faithfulness/empathy benchmark.

`no_rag_rules.json` — questions deliberately given *without* retrieved
context, each with a JSON-defined rule describing what acceptable model
behavior looks like when retrieval has nothing to offer (defer, hedge,
answer from base knowledge, etc). Mirrors the "what happens when RAG can't
cover every question" robustness check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GoldenQuestion:
    id: str
    category: str
    question: str
    reference_answer: str
    context: list[str] = field(default_factory=list)

    @property
    def requires_rag(self) -> bool:
        return bool(self.context)


@dataclass
class NoRagCase:
    id: str
    category: str
    question: str
    acceptable_behaviors: list[str]
    must_acknowledge_uncertainty: bool = False
    forbidden_behaviors: list[str] = field(default_factory=list)

    @property
    def expected_behavior(self) -> dict:
        """The behavior rule as stored alongside each result."""
        return {
            "acceptable_behaviors": self.acceptable_behaviors,
            "forbidden_behaviors": self.forbidden_behaviors,
            "must_acknowledge_uncertainty": self.must_acknowledge_uncertainty,
        }


def load_golden_dataset(path: str | Path) -> list[GoldenQuestion]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [GoldenQuestion(**item) for item in raw]


def load_no_rag_dataset(path: str | Path) -> list[NoRagCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [NoRagCase(**item) for item in raw]
