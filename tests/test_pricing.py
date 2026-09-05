"""Tests for the pricing module (Feature 2 — Dollar Cost Tracking)."""

from __future__ import annotations

import pytest

from longguard.config import GuardConfig
from longguard.core.breaker import CircuitBreaker
from longguard.core.step import AgentStep
from longguard.pricing import (
    PRICING_TABLE,
    compute_cost,
    list_supported_models,
    lookup,
)

# ===========================================================================
# lookup()
# ===========================================================================


class TestLookup:
    def test_known_model_returns_pricing(self) -> None:
        p = lookup("gpt-4o")
        assert p is not None
        assert "input" in p
        assert "output" in p

    def test_case_insensitive(self) -> None:
        assert lookup("GPT-4O") == lookup("gpt-4o")
        assert lookup("Claude-3-5-Sonnet") == lookup("claude-3-5-sonnet")

    def test_leading_trailing_whitespace(self) -> None:
        assert lookup("  gpt-4o  ") == lookup("gpt-4o")

    def test_unknown_model_returns_none(self) -> None:
        assert lookup("nonexistent-model-xyz-9999") is None

    def test_all_table_entries_have_positive_prices(self) -> None:
        for model, pricing in PRICING_TABLE.items():
            assert pricing["input"] > 0, f"{model}: input price must be > 0"
            assert pricing["output"] > 0, f"{model}: output price must be > 0"
            # Output is generally >= input
            assert pricing["output"] >= pricing["input"] / 10, (
                f"{model}: output price suspiciously low vs input"
            )

    def test_gpt4o_mini_cheaper_than_gpt4o(self) -> None:
        gpt4o = lookup("gpt-4o")
        mini = lookup("gpt-4o-mini")
        assert gpt4o is not None and mini is not None
        assert mini["input"] < gpt4o["input"]
        assert mini["output"] < gpt4o["output"]

    def test_claude_opus_more_expensive_than_haiku(self) -> None:
        opus = lookup("claude-3-opus")
        haiku = lookup("claude-3-haiku")
        assert opus is not None and haiku is not None
        assert opus["input"] > haiku["input"]

    def test_anthropic_dated_variants(self) -> None:
        assert lookup("claude-3-5-sonnet-20241022") is not None
        assert lookup("claude-3-haiku-20240307") is not None


# ===========================================================================
# compute_cost()
# ===========================================================================


class TestComputeCost:
    def test_user_supplied_prices_win(self) -> None:
        cost = compute_cost(
            model="gpt-4o",  # would resolve via table
            input_tokens=1000,
            output_tokens=500,
            cost_per_input_token=1e-4,   # deliberately very high
            cost_per_output_token=2e-4,
        )
        expected = 1000 * 1e-4 + 500 * 2e-4
        assert cost == pytest.approx(expected)

    def test_user_prices_no_model(self) -> None:
        """Pricing works without specifying a model when user supplies prices."""
        cost = compute_cost(
            model=None,
            input_tokens=100,
            output_tokens=50,
            cost_per_input_token=2e-6,
            cost_per_output_token=8e-6,
        )
        expected = 100 * 2e-6 + 50 * 8e-6
        assert cost == pytest.approx(expected)

    def test_table_lookup_gpt4o(self) -> None:
        cost = compute_cost("gpt-4o", 1000, 1000)
        # 1000 * 2.5e-6 + 1000 * 10e-6 = 0.0125
        assert cost == pytest.approx(0.0125)

    def test_table_lookup_gpt4o_mini(self) -> None:
        cost = compute_cost("gpt-4o-mini", 10_000, 2_000)
        expected = 10_000 * 0.15e-6 + 2_000 * 0.6e-6
        assert cost == pytest.approx(expected)

    def test_unknown_model_no_user_prices_returns_none(self) -> None:
        cost = compute_cost("mystery-model-v9", 1000, 500)
        assert cost is None

    def test_none_model_no_user_prices_returns_none(self) -> None:
        cost = compute_cost(None, 1000, 500)
        assert cost is None

    def test_zero_tokens_returns_zero(self) -> None:
        cost = compute_cost("gpt-4o", 0, 0)
        assert cost == pytest.approx(0.0)

    def test_only_input_tokens(self) -> None:
        cost = compute_cost("gpt-4o", 1000, 0)
        assert cost == pytest.approx(1000 * 2.5e-6)

    def test_only_output_tokens(self) -> None:
        cost = compute_cost("gpt-4o", 0, 1000)
        assert cost == pytest.approx(1000 * 10e-6)

    def test_claude_sonnet_pricing(self) -> None:
        cost = compute_cost("claude-3-5-sonnet", 500, 200)
        expected = 500 * 3e-6 + 200 * 15e-6
        assert cost == pytest.approx(expected)

    def test_gemini_flash_pricing(self) -> None:
        cost = compute_cost("gemini-1.5-flash", 50_000, 10_000)
        expected = 50_000 * 0.075e-6 + 10_000 * 0.3e-6
        assert cost == pytest.approx(expected)

    def test_user_price_overrides_table_when_both_provided(self) -> None:
        user_cost = compute_cost(
            model="gpt-4o",
            input_tokens=100,
            output_tokens=100,
            cost_per_input_token=0.001,
            cost_per_output_token=0.002,
        )
        table_cost = compute_cost("gpt-4o", 100, 100)
        assert user_cost != table_cost  # user price is wildly different
        assert user_cost == pytest.approx(0.001 * 100 + 0.002 * 100)


# ===========================================================================
# list_supported_models()
# ===========================================================================


class TestListSupportedModels:
    def test_returns_list_of_strings(self) -> None:
        models = list_supported_models()
        assert isinstance(models, list)
        assert all(isinstance(m, str) for m in models)

    def test_sorted(self) -> None:
        models = list_supported_models()
        assert models == sorted(models)

    def test_contains_common_models(self) -> None:
        models = list_supported_models()
        assert "gpt-4o" in models
        assert "claude-3-5-sonnet" in models
        assert "gemini-1.5-pro" in models

    def test_no_duplicates(self) -> None:
        models = list_supported_models()
        assert len(models) == len(set(models))

    def test_matches_pricing_table_keys(self) -> None:
        models = list_supported_models()
        assert set(models) == set(PRICING_TABLE.keys())


# ===========================================================================
# Integration: cost tracking inside CircuitBreaker
# ===========================================================================


class TestBreakerCostTracking:
    """Smoke tests confirming the breaker correctly tracks and reports cost."""

    def _make_step(self, n: int, tokens: int = 100) -> AgentStep:
        return AgentStep(
            step_number=n,
            thought=f"Step {n} thought",
            tokens_used=tokens,
        )

    def test_cost_appears_in_report_when_model_set(self) -> None:
        config = GuardConfig(model="gpt-4o")
        breaker = CircuitBreaker(config)
        breaker.check(self._make_step(1, tokens=1000))
        report = breaker.report
        assert report.estimated_cost_usd is not None
        assert report.estimated_cost_usd > 0
        assert report.model == "gpt-4o"

    def test_no_cost_when_no_model(self) -> None:
        config = GuardConfig()  # no model
        breaker = CircuitBreaker(config)
        breaker.check(self._make_step(1, tokens=1000))
        assert breaker.report.estimated_cost_usd is None

    def test_max_cost_usd_trips_kill(self) -> None:
        # max_cost_usd very low so one step blows it
        config = GuardConfig(model="gpt-4o", max_cost_usd=0.000001)
        breaker = CircuitBreaker(config)
        decision = breaker.check(self._make_step(1, tokens=10_000))
        assert decision.action == "kill"
        assert "cost_budget_exceeded" in decision.reason

    def test_cost_accumulates_across_steps(self) -> None:
        config = GuardConfig(model="gpt-4o-mini", max_steps=50)
        breaker = CircuitBreaker(config)
        for i in range(1, 6):
            breaker.check(self._make_step(i, tokens=200))
        report = breaker.report
        assert report.estimated_cost_usd is not None
        # 5 steps × 200 tokens × gpt-4o-mini price > 0
        assert report.estimated_cost_usd > 0

    def test_reset_clears_cost(self) -> None:
        config = GuardConfig(model="gpt-4o")
        breaker = CircuitBreaker(config)
        breaker.check(self._make_step(1, tokens=500))
        assert breaker.report.estimated_cost_usd is not None

        breaker.reset()
        assert breaker.report.estimated_cost_usd is None

    def test_cost_in_summary_string(self) -> None:
        config = GuardConfig(model="gpt-4o")
        breaker = CircuitBreaker(config)
        breaker.check(self._make_step(1, tokens=1000))
        summary = breaker.report.summary()
        assert "Estimated Cost" in summary
        assert "$" in summary
        assert "USD" in summary

    def test_user_supplied_prices_no_model(self) -> None:
        config = GuardConfig(
            cost_per_input_token=5e-6,
            cost_per_output_token=15e-6,
            max_cost_usd=10.0,
        )
        breaker = CircuitBreaker(config)
        breaker.check(self._make_step(1, tokens=100))
        assert breaker.report.estimated_cost_usd is not None
