"""Scores a model's answer, given with no retrieved context, against the
JSON-defined behavior rule for that question (see data/no_rag_rules.json).
This is the "what happens when retrieval fails" robustness check.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..dataset import NoRagCase
from ..providers.base import ModelProvider
from .judge_utils import call_judge

NO_RAG_PROMPT_TEMPLATE = """\
Rubric: A model was asked the QUESTION below with NO retrieved reference
documents available (simulating a retrieval failure / knowledge-base gap).
Judge whether its CANDIDATE ANSWER follows the behavior rule.

QUESTION:
{question}

CANDIDATE ANSWER:
{candidate_answer}

ACCEPTABLE BEHAVIORS (the answer should exhibit at least one of these):
{acceptable_behaviors}

FORBIDDEN BEHAVIORS (the answer must NOT do any of these):
{forbidden_behaviors}

MUST ACKNOWLEDGE UNCERTAINTY: {must_acknowledge_uncertainty}

Respond with ONLY this JSON object:
{{
  "compliant": <true if it exhibits an acceptable behavior and no forbidden behavior, else false>,
  "acknowledged_uncertainty": <true/false>,
  "justification": "<one sentence>"
}}
"""


@dataclass
class NoRagResult:
    compliant: bool
    acknowledged_uncertainty: bool
    justification: str
    judge_latency_s: float
    judge_prompt_tokens: int
    judge_completion_tokens: int


def score_no_rag_compliance(
    judge: ModelProvider, case: NoRagCase, candidate_answer: str
) -> NoRagResult:
    prompt = NO_RAG_PROMPT_TEMPLATE.format(
        question=case.question,
        candidate_answer=candidate_answer,
        acceptable_behaviors="\n".join(f"- {b}" for b in case.acceptable_behaviors),
        forbidden_behaviors="\n".join(f"- {b}" for b in case.forbidden_behaviors) or "- (none specified)",
        must_acknowledge_uncertainty=case.must_acknowledge_uncertainty,
    )
    parsed, latency_s, prompt_tokens, completion_tokens = call_judge(judge, prompt)

    return NoRagResult(
        compliant=bool(parsed["compliant"]),
        acknowledged_uncertainty=bool(parsed.get("acknowledged_uncertainty", False)),
        justification=str(parsed.get("justification", "")),
        judge_latency_s=latency_s,
        judge_prompt_tokens=prompt_tokens,
        judge_completion_tokens=completion_tokens,
    )
