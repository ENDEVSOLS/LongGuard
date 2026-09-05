"""LongGuard model pricing registry.

Provides a best-effort pricing table for common LLM models so that
GuardReport can estimate dollar cost without any external API calls.

Pricing Resolution Tiers
------------------------
1. **User-supplied** (highest priority): Set ``cost_per_input_token`` and
   ``cost_per_output_token`` in :class:`~longguard.config.GuardConfig`.
2. **Built-in table**: This module — a snapshot of provider list prices at
   the time of the LongGuard release.  Updated each minor release.
3. **Graceful fallback**: If the model is not recognised and no per-token
   prices are supplied, ``estimated_cost_usd`` will be ``None`` in the
   report and any ``max_cost_usd`` cap will be silently skipped.

Prices are in **USD per token** (not per 1 000 tokens).

Last updated: 2026-09-05 (LongGuard v0.1.3)
"""

from __future__ import annotations

from typing import TypedDict


class ModelPricing(TypedDict):
    """Per-token pricing for a single model."""

    input: float   # USD per input token
    output: float  # USD per output token


# ---------------------------------------------------------------------------
# Built-in pricing table
# Keys are normalised model identifiers.  The lookup is case-insensitive and
# strips leading/trailing whitespace.  For models with date suffixes (e.g.
# "gpt-4o-2024-11-20") we also add the base name as an alias so either form
# resolves correctly.
#
# Source: provider list prices as of 2026-09-05.
# Always verify current prices at:
#   https://openai.com/api/pricing/
#   https://www.anthropic.com/pricing#anthropic-api
#   https://ai.google.dev/pricing
# ---------------------------------------------------------------------------

PRICING_TABLE: dict[str, ModelPricing] = {
    # ── OpenAI ───────────────────────────────────────────────────────────
    "gpt-4o": {"input": 2.5e-6, "output": 10e-6},
    "gpt-4o-mini": {"input": 0.15e-6, "output": 0.6e-6},
    "gpt-4o-mini-2024-07-18": {"input": 0.15e-6, "output": 0.6e-6},
    "gpt-4-turbo": {"input": 10e-6, "output": 30e-6},
    "gpt-4-turbo-preview": {"input": 10e-6, "output": 30e-6},
    "gpt-4": {"input": 30e-6, "output": 60e-6},
    "gpt-3.5-turbo": {"input": 0.5e-6, "output": 1.5e-6},
    "o1": {"input": 15e-6, "output": 60e-6},
    "o1-mini": {"input": 3e-6, "output": 12e-6},
    "o3": {"input": 10e-6, "output": 40e-6},
    "o3-mini": {"input": 1.1e-6, "output": 4.4e-6},
    "o4-mini": {"input": 1.1e-6, "output": 4.4e-6},
    # ── Anthropic ────────────────────────────────────────────────────────
    "claude-3-5-sonnet": {"input": 3e-6, "output": 15e-6},
    "claude-3-5-sonnet-20241022": {"input": 3e-6, "output": 15e-6},
    "claude-3-5-haiku": {"input": 0.8e-6, "output": 4e-6},
    "claude-3-5-haiku-20241022": {"input": 0.8e-6, "output": 4e-6},
    "claude-3-opus": {"input": 15e-6, "output": 75e-6},
    "claude-3-opus-20240229": {"input": 15e-6, "output": 75e-6},
    "claude-3-sonnet": {"input": 3e-6, "output": 15e-6},
    "claude-3-haiku": {"input": 0.25e-6, "output": 1.25e-6},
    "claude-3-haiku-20240307": {"input": 0.25e-6, "output": 1.25e-6},
    "claude-sonnet-4": {"input": 3e-6, "output": 15e-6},
    "claude-opus-4": {"input": 15e-6, "output": 75e-6},
    "claude-haiku-4": {"input": 0.8e-6, "output": 4e-6},
    # ── Google ───────────────────────────────────────────────────────────
    "gemini-1.5-pro": {"input": 1.25e-6, "output": 5e-6},
    "gemini-1.5-flash": {"input": 0.075e-6, "output": 0.3e-6},
    "gemini-1.5-flash-8b": {"input": 0.0375e-6, "output": 0.15e-6},
    "gemini-2.0-flash": {"input": 0.1e-6, "output": 0.4e-6},
    "gemini-2.0-flash-lite": {"input": 0.075e-6, "output": 0.3e-6},
    "gemini-2.5-pro": {"input": 1.25e-6, "output": 10e-6},
    "gemini-2.5-flash": {"input": 0.075e-6, "output": 0.3e-6},
    # ── Meta / Groq / Together (popular hosted Llama) ────────────────────
    "llama-3.1-8b-instant": {"input": 0.05e-6, "output": 0.08e-6},
    "llama-3.1-70b-versatile": {"input": 0.59e-6, "output": 0.79e-6},
    "llama-3.3-70b-versatile": {"input": 0.59e-6, "output": 0.79e-6},
    "llama-3.1-405b-reasoning": {"input": 2.0e-6, "output": 2.0e-6},
    # ── Mistral ──────────────────────────────────────────────────────────
    "mistral-large": {"input": 2e-6, "output": 6e-6},
    "mistral-small": {"input": 0.1e-6, "output": 0.3e-6},
    "mistral-nemo": {"input": 0.13e-6, "output": 0.13e-6},
    "codestral": {"input": 0.2e-6, "output": 0.6e-6},
    # ── Cohere ───────────────────────────────────────────────────────────
    "command-r-plus": {"input": 2.5e-6, "output": 10e-6},
    "command-r": {"input": 0.15e-6, "output": 0.6e-6},
}


def _normalise(model: str) -> str:
    """Normalise a model identifier for table lookup."""
    return model.strip().lower()


def lookup(model: str) -> ModelPricing | None:
    """Look up pricing for a model.

    The lookup is case-insensitive.  Returns ``None`` if the model is not
    in the built-in table (use user-supplied prices in that case).

    Args:
        model: Model identifier, e.g. ``"gpt-4o"`` or ``"claude-3-5-sonnet"``.

    Returns:
        A :class:`ModelPricing` dict with ``"input"`` and ``"output"`` per-token
        prices in USD, or ``None`` if the model is not recognised.

    Example::

        from longguard.pricing import lookup
        pricing = lookup("gpt-4o")
        # {"input": 2.5e-06, "output": 1e-05}
    """
    return PRICING_TABLE.get(_normalise(model))


def compute_cost(
    model: str | None,
    input_tokens: int,
    output_tokens: int,
    *,
    cost_per_input_token: float | None = None,
    cost_per_output_token: float | None = None,
) -> float | None:
    """Compute estimated USD cost for a generation.

    Resolution order:
    1. ``cost_per_input_token`` / ``cost_per_output_token`` — if both supplied,
       they win unconditionally.
    2. Built-in ``PRICING_TABLE`` lookup by ``model``.
    3. ``None`` — returned when pricing cannot be determined.

    Args:
        model: Model identifier (used for table lookup). May be ``None``.
        input_tokens: Number of input tokens in the generation.
        output_tokens: Number of output tokens in the generation.
        cost_per_input_token: Override price per input token (USD).
        cost_per_output_token: Override price per output token (USD).

    Returns:
        Estimated cost in USD, or ``None`` if pricing is unavailable.

    Example::

        cost = compute_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        # 0.0075 (= 1000 * 2.5e-6 + 500 * 10e-6)
    """
    # Tier 1: user-supplied prices
    if cost_per_input_token is not None and cost_per_output_token is not None:
        return (
            input_tokens * cost_per_input_token
            + output_tokens * cost_per_output_token
        )

    # Tier 2: built-in table
    if model is not None:
        pricing = lookup(model)
        if pricing is not None:
            return (
                input_tokens * pricing["input"]
                + output_tokens * pricing["output"]
            )

    # Tier 3: graceful fallback
    return None


def list_supported_models() -> list[str]:
    """Return a sorted list of model identifiers in the built-in pricing table.

    Returns:
        Sorted list of supported model identifier strings.

    Example::

        from longguard.pricing import list_supported_models
        models = list_supported_models()
        # ['claude-3-5-haiku', 'claude-3-5-haiku-20241022', ...]
    """
    return sorted(PRICING_TABLE.keys())
