"""Cost modeling. Local Ollama/Hugging Face models are $0, but the same
formula applies to a hosted API model once its $/1K-token rates are filled
into the config — useful for comparing local vs. hosted cost tradeoffs.
"""

from __future__ import annotations


def compute_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    cost_per_1k_prompt_tokens: float,
    cost_per_1k_completion_tokens: float,
) -> float:
    return (
        (prompt_tokens / 1000) * cost_per_1k_prompt_tokens
        + (completion_tokens / 1000) * cost_per_1k_completion_tokens
    )
