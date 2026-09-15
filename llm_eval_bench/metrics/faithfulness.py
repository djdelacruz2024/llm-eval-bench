"""LLM-as-judge faithfulness/grounding scoring: is every claim in the
candidate answer supported by the retrieved context, or did the model add
unsupported claims (a proxy for hallucination)?
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..providers.base import ModelProvider
from .judge_utils import call_judge

FAITHFULNESS_PROMPT_TEMPLATE = """\
Rubric: Determine whether the CANDIDATE ANSWER is fully grounded in the
CONTEXT passages, or whether it makes claims the context does not support
(unsupported claims are a sign of hallucination).

CONTEXT:
{context}

CANDIDATE ANSWER:
{candidate_answer}

Respond with ONLY this JSON object:
{{
  "faithfulness_score": <float 0.0-1.0, fraction of claims that ARE supported by context>,
  "unsupported_claims": [<list of short strings, each an unsupported claim; empty list if none>],
  "justification": "<one sentence>"
}}
"""


@dataclass
class FaithfulnessResult:
    faithfulness_score: float
    unsupported_claims: list[str] = field(default_factory=list)
    justification: str = ""
    judge_latency_s: float = 0.0
    judge_prompt_tokens: int = 0
    judge_completion_tokens: int = 0

    @property
    def has_hallucination(self) -> bool:
        return len(self.unsupported_claims) > 0


def score_faithfulness(
    judge: ModelProvider, context: list[str], candidate_answer: str
) -> FaithfulnessResult:
    if not context:
        # No retrieval context was provided at all (e.g. the no-RAG suite) —
        # faithfulness isn't a meaningful concept without something to be
        # faithful to, so the caller should use the no-RAG compliance metric
        # instead of this one.
        raise ValueError("score_faithfulness requires non-empty context")

    prompt = FAITHFULNESS_PROMPT_TEMPLATE.format(
        context="\n".join(f"- {c}" for c in context), candidate_answer=candidate_answer
    )
    parsed, latency_s, prompt_tokens, completion_tokens = call_judge(judge, prompt)

    return FaithfulnessResult(
        faithfulness_score=float(parsed["faithfulness_score"]),
        unsupported_claims=list(parsed.get("unsupported_claims", [])),
        justification=str(parsed.get("justification", "")),
        judge_latency_s=latency_s,
        judge_prompt_tokens=prompt_tokens,
        judge_completion_tokens=completion_tokens,
    )
