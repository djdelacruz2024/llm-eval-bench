from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from llm_eval_bench.providers.mock_provider import MockJudgeProvider, MockProvider  # noqa: E402


@pytest.fixture
def judge() -> MockJudgeProvider:
    return MockJudgeProvider(name="judge")


@pytest.fixture
def mock_model() -> MockProvider:
    return MockProvider(name="mock-a", canned_response="This is a stable test answer.")


@pytest.fixture
def golden_dataset_path() -> str:
    return str(REPO_ROOT / "data" / "golden_dataset.json")


@pytest.fixture
def no_rag_dataset_path() -> str:
    return str(REPO_ROOT / "data" / "no_rag_rules.json")


@pytest.fixture
def demo_config_path() -> str:
    return str(REPO_ROOT / "config.demo.yaml")
