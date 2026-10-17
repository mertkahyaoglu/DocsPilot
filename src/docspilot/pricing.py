"""Token pricing (USD per 1M tokens). Cached from provider docs: re-check before trusting
absolute numbers. Used for relative comparisons (prompt A vs B) and budget guards."""

from __future__ import annotations

from dataclasses import dataclass

from docspilot.llm.types import Usage


@dataclass(frozen=True, slots=True)
class Price:
    input: float
    output: float


# Prompt cache: reads ~0.1x input price, 5-minute cache writes ~1.25x.
CACHE_READ_MULT = 0.1
CACHE_WRITE_MULT = 1.25

PRICES: dict[str, Price] = {
    "claude-opus-5-5": Price(4.0, 20.0),
    "claude-opus-5": Price(5.0, 25.0),
    "claude-sonnet-5-5": Price(2.0, 10.0),
    "claude-sonnet-5": Price(2.0, 10.0),
    "claude-haiku-5-5": Price(0.10, 0.50),
    "claude-fable-5-1": Price(10.0, 50.0),
    # Cloudflare Workers AI (USD per 1M tokens, from the models API at the time of writing)
    "@cf/openai/gpt-oss-120b": Price(0.35, 0.75),
    "@cf/openai/gpt-oss-20b": Price(0.20, 0.30),
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast": Price(0.293, 2.253),
    "@cf/meta/llama-4-scout-17b-16e-instruct": Price(0.27, 0.85),
    "@cf/mistralai/mistral-small-3.1-24b-instruct": Price(0.351, 0.555),
    "@cf/google/gemma-4-26b-a4b-it": Price(0.10, 0.30),
    "@cf/zai-org/glm-4.7-flash": Price(0.0605, 0.40),
    "@cf/qwen/qwen3-30b-a3b-fp8": Price(0.0509, 0.335),
}


def cost_usd(model: str, usage: Usage) -> float | None:
    """None for unknown models: never report a made-up number."""
    price = PRICES.get(model)
    if price is None:
        return None
    per_m = 1_000_000
    return (
        usage.input_tokens * price.input
        + usage.cache_read_input_tokens * price.input * CACHE_READ_MULT
        + usage.cache_creation_input_tokens * price.input * CACHE_WRITE_MULT
        + usage.output_tokens * price.output
    ) / per_m


def cost_or_ceiling(model: str, usage: Usage) -> float:
    """For *limits*: an unpriced model is charged at the most expensive known rate, so a
    new model id can never be a way around a budget."""
    exact = cost_usd(model, usage)
    if exact is not None:
        return exact
    worst = max(PRICES.values(), key=lambda p: p.input + p.output)
    per_m = 1_000_000
    return (
        (usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens)
        * worst.input
        + usage.output_tokens * worst.output
    ) / per_m
