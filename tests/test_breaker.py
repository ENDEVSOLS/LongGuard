"""Tests for CircuitBreaker state machine."""

import pytest

from longguard.config import GuardConfig
from longguard.core.breaker import BreakerDecision, BreakerState, CircuitBreaker
from longguard.core.step import AgentStep


class TestBreakerStateClosed:
    """Test behavior in CLOSED (normal) state."""

    def test_clean_steps_continue(self, make_step):
        """Clean steps return continue decision."""
        breaker = CircuitBreaker(GuardConfig())
        step = make_step()
        decision = breaker.check(step)
        assert decision.action == "continue"
        assert breaker.state == BreakerState.CLOSED

    def test_diverse_steps_stay_closed(self, diverse_steps):
        """Diverse steps keep the breaker in CLOSED state."""
        breaker = CircuitBreaker(GuardConfig())
        for step in diverse_steps:
            decision = breaker.check(step)
            assert decision.action == "continue"
        assert breaker.state == BreakerState.CLOSED


class TestBreakerDetectionToReflection:
    """Test transition from CLOSED to REFLECTING on first detection."""

    def test_tool_repeat_triggers_reflect(self, make_step):
        """Tool repeat detection triggers reflect action."""
        config = GuardConfig(tool_repeat_threshold=2, tool_repeat_window=10)
        breaker = CircuitBreaker(config)

        # Feed identical tool calls — step 2 should trigger reflect
        decisions = []
        for i in range(3):
            step = AgentStep(
                step_number=i + 1,
                thought=f"Try {i}",
                action="search",
                action_input="same query",
                observation="No results",
                tokens_used=50,
            )
            decision = breaker.check(step)
            decisions.append(decision)

        # At some point we should have gotten a reflect decision
        reflect_decisions = [d for d in decisions if d.action == "reflect"]
        assert len(reflect_decisions) >= 1
        assert reflect_decisions[0].inject_prompt is not None
        assert "LOOP DETECTED" in reflect_decisions[0].inject_prompt
        # After the first reflect, breaker should be in REFLECTING
        # After subsequent identical calls, it may transition further
        assert breaker.state in (BreakerState.REFLECTING, BreakerState.HALF_OPEN, BreakerState.OPEN)

    def test_reflect_decision_has_reason(self, make_step):
        """Reflect decision includes the pattern name as reason."""
        config = GuardConfig(tool_repeat_threshold=2, tool_repeat_window=10)
        breaker = CircuitBreaker(config)

        decisions = []
        for i in range(3):
            step = AgentStep(
                step_number=i + 1,
                thought=f"Try {i}",
                action="search",
                action_input="same query",
                tokens_used=50,
            )
            decision = breaker.check(step)
            decisions.append(decision)

        # The first reflect decision should have the pattern name
        reflect_decisions = [d for d in decisions if d.action == "reflect"]
        assert len(reflect_decisions) >= 1
        assert "tool_repeat" in reflect_decisions[0].reason


class TestBreakerReflectionToHalfOpen:
    """Test transition from REFLECTING to HALF_OPEN."""

    def test_persistent_loop_after_reflect(self, make_step):
        """Loop persisting after reflection moves to HALF_OPEN then OPEN."""
        config = GuardConfig(
            tool_repeat_threshold=2,
            tool_repeat_window=10,
            max_reflections=2,
            max_tokens_per_run=1_000_000,
            max_steps=100,
        )
        breaker = CircuitBreaker(config)

        # Feed enough identical calls to trigger reflection, then keep going
        decisions = []
        for i in range(8):
            step = AgentStep(
                step_number=i + 1,
                thought=f"Try {i}",
                action="search",
                action_input="same query",
                tokens_used=50,
            )
            decision = breaker.check(step)
            decisions.append(decision)

        # Should have reflect, then eventually kill
        actions = [d.action for d in decisions]
        assert "reflect" in actions
        assert "kill" in actions
        assert breaker.state == BreakerState.OPEN


class TestBreakerKill:
    """Test kill decisions."""

    def test_token_budget_kill(self):
        """Exceeding token budget triggers kill."""
        config = GuardConfig(max_tokens_per_run=500)
        breaker = CircuitBreaker(config)

        step = AgentStep(
            step_number=1,
            thought="Big step",
            tokens_used=600,
        )
        decision = breaker.check(step)
        assert decision.action == "kill"
        assert "token_budget_exceeded" in decision.reason
        assert breaker.state == BreakerState.OPEN

    def test_max_steps_kill(self):
        """Exceeding step count triggers kill."""
        config = GuardConfig(max_steps=3, max_tokens_per_run=1_000_000)
        breaker = CircuitBreaker(config)

        for i in range(4):
            step = AgentStep(
                step_number=i + 1,
                thought=f"Step {i}",
                tokens_used=10,
            )
            decision = breaker.check(step)

        assert decision.action == "kill"
        assert "max_steps_exceeded" in decision.reason

    def test_kill_has_report(self, make_step):
        """Kill decision includes a GuardReport."""
        config = GuardConfig(max_tokens_per_run=50)
        breaker = CircuitBreaker(config)

        step = make_step(tokens_used=100)
        decision = breaker.check(step)

        assert decision.report is not None
        assert decision.report.kill_reason is not None


class TestBreakerRecovery:
    """Test recovery from REFLECTING back to CLOSED."""

    def test_clean_step_after_reflect(self):
        """A clean step after reflection recovers the breaker."""
        config = GuardConfig(
            tool_repeat_threshold=2,
            tool_repeat_window=10,
            max_tokens_per_run=1_000_000,
            max_steps=100,
        )
        breaker = CircuitBreaker(config)

        # Trigger reflection
        for i in range(3):
            step = AgentStep(
                step_number=i + 1,
                thought=f"Repeat {i}",
                action="search",
                action_input="same",
                tokens_used=50,
            )
            decision = breaker.check(step)

        assert decision.action == "reflect"

        # Now feed a diverse step (different tool)
        step = AgentStep(
            step_number=4,
            thought="Let me try a different approach.",
            action="calculator",
            action_input="2+2",
            observation="4",
            tokens_used=80,
        )
        decision = breaker.check(step)
        assert decision.action == "continue"

        # After a clean step in HALF_OPEN, we should go to CLOSED
        step = AgentStep(
            step_number=5,
            thought="And another different step.",
            action="lookup",
            action_input="info",
            observation="Data",
            tokens_used=60,
        )
        decision = breaker.check(step)
        assert decision.action == "continue"
        assert breaker.state == BreakerState.CLOSED


class TestBreakerEvents:
    """Test event emission."""

    def test_event_callback_called(self):
        """Event callback is called for detection, reflect, and kill events."""
        events = []

        def callback(event_name, data):
            events.append((event_name, data))

        config = GuardConfig(max_tokens_per_run=50)
        breaker = CircuitBreaker(config, event_callback=callback)

        step = AgentStep(step_number=1, thought="test", tokens_used=100)
        breaker.check(step)

        assert len(events) >= 1
        assert events[0][0] == "kill"

    def test_event_callback_exception_handled(self):
        """Exceptions in event callbacks are caught and logged."""
        def bad_callback(event_name, data):
            raise RuntimeError("Callback error")

        config = GuardConfig(max_tokens_per_run=5000)
        breaker = CircuitBreaker(config, event_callback=bad_callback)

        # Should not crash
        step = AgentStep(step_number=1, thought="test", tokens_used=50)
        decision = breaker.check(step)
        assert decision.action == "continue"


class TestBreakerReport:
    """Test GuardReport integration."""

    def test_report_tracks_tokens(self):
        """Report accurately tracks total tokens."""
        config = GuardConfig(max_tokens_per_run=1_000_000)
        breaker = CircuitBreaker(config)

        for i in range(5):
            step = AgentStep(
                step_number=i + 1,
                thought="thinking",
                tokens_used=100,
            )
            breaker.check(step)

        assert breaker.report.total_tokens == 500
        assert breaker.total_tokens == 500

    def test_report_tracks_steps(self):
        """Report accurately tracks total steps."""
        config = GuardConfig(max_tokens_per_run=1_000_000, max_steps=100)
        breaker = CircuitBreaker(config)

        for i in range(5):
            step = AgentStep(
                step_number=i + 1,
                thought="thinking",
                tokens_used=10,
            )
            breaker.check(step)

        assert breaker.total_steps == 5


class TestBreakerReset:
    """Test the reset method."""

    def test_reset_returns_to_closed(self):
        """After reset, breaker is in CLOSED state."""
        config = GuardConfig(max_tokens_per_run=50)
        breaker = CircuitBreaker(config)

        # Kill the breaker
        step = AgentStep(step_number=1, thought="test", tokens_used=100)
        breaker.check(step)
        assert breaker.state == BreakerState.OPEN

        # Reset
        breaker.reset()
        assert breaker.state == BreakerState.CLOSED
        assert breaker.total_tokens == 0
        assert breaker.total_steps == 0
        assert breaker.is_open is False


class TestBreakerAddDetector:
    """Test custom detector registration."""

    def test_add_custom_detector(self, make_step):
        """Can add a custom detector at runtime."""
        from longguard.core.detectors.base import AbstractLoopDetector, DetectionResult

        class AlwaysDetect(AbstractLoopDetector):
            @property
            def name(self):
                return "AlwaysDetect"

            def analyze(self, step):
                return DetectionResult(
                    detected=True, pattern="custom", confidence=0.9, evidence={}
                )

            def reset(self):
                pass

        config = GuardConfig(max_tokens_per_run=1_000_000, max_steps=100)
        breaker = CircuitBreaker(config)
        breaker.add_detector(AlwaysDetect())

        step = make_step()
        decision = breaker.check(step)
        assert decision.action == "reflect"


class TestBreakerDecision:
    """Test BreakerDecision dataclass."""

    def test_invalid_action_raises(self):
        """Invalid action raises ValueError."""
        with pytest.raises(ValueError, match="Invalid breaker action"):
            BreakerDecision(action="invalid")

    def test_bool_continue_is_true(self):
        """Continue decisions are truthy."""
        decision = BreakerDecision(action="continue")
        assert bool(decision) is True

    def test_bool_kill_is_false(self):
        """Kill decisions are falsy."""
        decision = BreakerDecision(action="kill")
        assert bool(decision) is False

    def test_bool_reflect_is_true(self):
        """Reflect decisions are truthy (agent should proceed after injection)."""
        decision = BreakerDecision(action="reflect", inject_prompt="pivot")
        assert bool(decision) is True
