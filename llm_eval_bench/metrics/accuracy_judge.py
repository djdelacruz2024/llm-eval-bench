"""LLM-as-judge accuracy scoring: does the candidate answer correctly convey
the same information as the reference answer?
"""

from __future__ import annotations

from dataclasses import dataclass

from ..providers.base import ModelProvider
from .judge_utils import call_judge

ACCURACY_PROMPT_TEMPLATE = """\
Rubric: Score how accurately the CANDIDATE ANSWER conveys the same factual
content as the REFERENCE ANSWER for the given QUESTION.

Score 1-5:
  5 = Fully correct, matches all key facts in the reference answer.
  4 = Correct on all major points, minor omission or imprecision.
  3 = Partially correct; misses or slightly misstates a key point.
  2 = Mostly incorrect; only a small part aligns with the reference.
  1 = Incorrect or contradicts the reference answer.

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

CANDIDATE ANSWER:
{candidate_answer}

Respond with ONLY this JSON object:
{{"score": <integer 1-5>, "correct": <true if score >= 4 else false>, "justification": "<one sentence>"}}
"""


@dataclass
class AccuracyResult:
    score: int
    correct: bool
    justification: str
    judge_latency_s: float
    judge_prompt_tokens: int
    judge_completion_tokens: int


def score_accuracy(
    judge: ModelProvider, question: str, reference_answer: str, candidate_answer: str
) -> AccuracyResult:
    prompt = ACCURACY_PROMPT_TEMPLATE.format(
        question=question, reference_answer=reference_answer, candidate_answer=candidate_answer
    )
    parsed, latency_s, prompt_tokens, completion_tokens = call_judge(judge, prompt)

    score = int(parsed["score"])
    return AccuracyResult(
        score=score,
        correct=bool(parsed.get("correct", score >= 4)),
        justification=str(parsed.get("justification", "")),
        judge_latency_s=latency_s,
        judge_prompt_tokens=prompt_tokens,
        judge_completion_tokens=completion_tokens,
    )
