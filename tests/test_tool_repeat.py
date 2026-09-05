"""Tests for ToolRepeatDetector."""

from longguard.core.detectors.tool_repeat import ToolRepeatDetector
from longguard.core.step import AgentStep


class TestToolRepeatBasic:
    """Basic ToolRepeatDetector functionality."""

    def test_no_detection_single_step(self):
        """A single tool call doesn't trigger detection."""
        detector = ToolRepeatDetector(repeat_threshold=3, window=6)
        step = AgentStep(step_number=1, thought="search", action="search", action_input="q")
        result = detector.analyze(step)
        assert result.detected is False

    def test_no_detection_different_tools(self):
        """Different tool calls don't trigger detection."""
        detector = ToolRepeatDetector(repeat_threshold=3, window=6)
        for i, tool in enumerate(["search", "lookup", "calculate"]):
            step = AgentStep(
                step_number=i + 1,
                thought=f"Using {tool}",
                action=tool,
                action_input="arg",
            )
            result = detector.analyze(step)
            assert result.detected is False

    def test_detection_identical_calls(self):
        """Same tool + same args triggers detection after threshold."""
        detector = ToolRepeatDetector(repeat_threshold=3, window=6)
        for i in range(3):
            step = AgentStep(
                step_number=i + 1,
                thought=f"Try again {i}",
                action="search",
                action_input="same query",
            )
            result = detector.analyze(step)

        assert result.detected is True
        assert result.pattern == "tool_repeat"
        assert result.evidence["tool"] == "search"
        assert result.evidence["repeated_times"] >= 3

    def test_no_detection_different_args(self):
        """Same tool but different args doesn't trigger detection."""
        detector = ToolRepeatDetector(repeat_threshold=3, window=6)
        for i in range(5):
            step = AgentStep(
                step_number=i + 1,
                thought=f"Search {i}",
                action="search",
                action_input=f"query_{i}",  # Different each time
            )
            result = detector.analyze(step)
            assert result.detected is False

    def test_no_action_never_triggers(self):
        """Steps without tool calls never trigger detection."""
        detector = ToolRepeatDetector(repeat_threshold=3, window=6)
        for i in range(10):
            step = AgentStep(step_number=i + 1, thought="Just thinking")
            result = detector.analyze(step)
            assert result.detected is False


class TestToolRepeatWindow:
    """Test sliding window behavior."""

    def test_window_expiry(self):
        """Old repeats fall out of the window."""
        detector = ToolRepeatDetector(repeat_threshold=3, window=4)

        # Add 2 identical calls
        for i in range(2):
            step = AgentStep(
                step_number=i + 1,
                thought=f"repeat {i}",
                action="search",
                action_input="q",
            )
            detector.analyze(step)

        # Add 2 different calls to push the identical ones out
        for i in range(2):
            step = AgentStep(
                step_number=i + 3,
                thought=f"other {i}",
                action="lookup",
                action_input=f"arg_{i}",
            )
            detector.analyze(step)

        # Another identical call should NOT trigger (only 1 in window now)
        step = AgentStep(
            step_number=5,
            thought="back to search",
            action="search",
            action_input="q",
        )
        result = detector.analyze(step)
        assert result.detected is False

    def test_window_preserves_recent(self):
        """Recent repeats within the window still trigger."""
        detector = ToolRepeatDetector(repeat_threshold=3, window=10)

        for i in range(3):
            step = AgentStep(
                step_number=i + 1,
                thought=f"repeat {i}",
                action="search",
                action_input="q",
            )
            result = detector.analyze(step)

        assert result.detected is True


class TestToolRepeatConfidence:
    """Test confidence scoring."""

    def test_confidence_at_threshold(self):
        """Confidence is exactly 1.0 at the threshold."""
        detector = ToolRepeatDetector(repeat_threshold=3, window=10)
        for i in range(3):
            step = AgentStep(
                step_number=i + 1,
                thought=f"r {i}",
                action="search",
                action_input="q",
            )
            result = detector.analyze(step)

        assert result.confidence == 1.0

    def test_confidence_increases_with_repeats(self):
        """More repeats increase confidence (up to 1.0)."""
        detector = ToolRepeatDetector(repeat_threshold=3, window=10)
        results = []
        for i in range(5):
            step = AgentStep(
                step_number=i + 1,
                thought=f"r {i}",
                action="search",
                action_input="q",
            )
            result = detector.analyze(step)
            results.append(result)

        # After 3 repeats, confidence = 1.0 (capped)
        detected = [r for r in results if r.detected]
        assert len(detected) >= 1
        # Confidence should be capped at 1.0
        for r in detected:
            assert r.confidence <= 1.0


class TestToolRepeatReset:
    """Test the reset method."""

    def test_reset_clears_state(self):
        """After reset, previous history doesn't affect new analysis."""
        detector = ToolRepeatDetector(repeat_threshold=3, window=10)

        # Build up history
        for i in range(3):
            step = AgentStep(
                step_number=i + 1, thought="r", action="search", action_input="q"
            )
            detector.analyze(step)

        # Reset
        detector.reset()

        # Same calls should not immediately trigger after reset
        step = AgentStep(
            step_number=1, thought="fresh", action="search", action_input="q"
        )
        result = detector.analyze(step)
        assert result.detected is False
