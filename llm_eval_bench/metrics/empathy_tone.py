"""Custom LLM-as-judge metric scoring tone/empathy — relevant for any
conversational support AI, where a factually correct answer delivered
coldly or dismissively is still a bad response.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..providers.base import ModelProvider
from .judge_utils import call_judge

EMPATHY_PROMPT_TEMPLATE = """\
Rubric: Score the tone of the CANDIDATE ANSWER for warmth, empathy, and
appropriateness, as if it were spoken to a customer asking a support or
general-knowledge question. The answer should be clear and respectful
without being cold, clinical-sounding, dismissive, or alarmist.

Score 1-5:
  5 = Warm, respectful, reassuring where appropriate, clearly written for a person (not a textbook).
  4 = Generally warm and appropriate, minor stiffness.
  3 = Neutral/clinical tone; not unkind but not warm either.
  2 = Noticeably cold, terse, or slightly dismissive.
  1 = Dismissive, alarmist, or inappropriate in tone.

QUESTION:
{question}

CANDIDATE ANSWER:
{candidate_answer}

Respond with ONLY this JSON object:
{{"score": <integer 1-5>, "justification": "<one sentence>"}}
"""


@dataclass
class EmpathyResult:
    score: int
    justification: str
    judge_latency_s: float
    judge_prompt_tokens: int
    judge_completion_tokens: int


def score_empathy(judge: ModelProvider, question: str, candidate_answer: str) -> EmpathyResult:
    prompt = EMPATHY_PROMPT_TEMPLATE.format(question=question, candidate_answer=candidate_answer)
    parsed, latency_s, prompt_tokens, completion_tokens = call_judge(judge, prompt)

    return EmpathyResult(
        score=int(parsed["score"]),
        justification=str(parsed.get("justification", "")),
        judge_latency_s=latency_s,
        judge_prompt_tokens=prompt_tokens,
        judge_completion_tokens=completion_tokens,
    )
