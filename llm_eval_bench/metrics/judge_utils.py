"""Shared plumbing for LLM-as-judge metrics: build a rubric prompt, call the
judge model, and robustly parse the JSON it (hopefully) returned.

Judge models don't always return clean JSON, even when asked to — they
wrap it in prose or markdown fences. `parse_judge_json` extracts the first
top-level `{...}` block rather than assuming the whole response is JSON.
"""

from __future__ import annotations

import json
import re

from ..providers.base import ModelProvider

JUDGE_SYSTEM_PROMPT = (
    "You are a strict, impartial evaluator of AI assistant responses. "
    "Follow the rubric exactly and respond with ONLY a single JSON object — "
    "no markdown fences, no commentary before or after it."
)


class JudgeParseError(ValueError):
    """Raised when the judge's response contains no parseable JSON object."""


def parse_judge_json(raw_text: str) -> dict:
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise JudgeParseError(f"No JSON object found in judge response: {raw_text!r}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise JudgeParseError(f"Judge response was not valid JSON: {raw_text!r}") from exc


def call_judge(judge: ModelProvider, rubric_prompt: str) -> tuple[dict, float, int, int]:
    """Runs the rubric prompt through the judge and returns
    (parsed_json, latency_s, prompt_tokens, completion_tokens).
    """
    result = judge.generate(rubric_prompt, system=JUDGE_SYSTEM_PROMPT)
    parsed = parse_judge_json(result.text)
    return parsed, result.latency_s, result.prompt_tokens, result.completion_tokens
