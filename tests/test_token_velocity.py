"""Tests for TokenVelocityDetector."""

from longguard.core.detectors.token_velocity import TokenVelocityDetector
from longguard.core.step import AgentStep


class TestTokenVelocityWarmup:
    """Test warmup period behavior."""

    def test_no_detection_during_warmup(self):
        """Detector doesn't trigger during warmup period."""
        detector = TokenVelocityDetector(warmup=3, velocity_multiplier=3.0)
        # Even with huge tokens
        for i in range(3):
            step = AgentStep(
                step_number=i + 1,
                thought="thinking",
                tokens_used=99999,
            )
            result = detector.analyze(step)
            assert result.detected is False

    def test_baseline_established_after_warmup(self):
        """Baseline is established after warmup steps."""
        detector = TokenVelocityDetector(warmup=3, velocity_multiplier=3.0)
        for i in range(3):
            step = AgentStep(
                step_number=i + 1,
                thought="thinking",
                tokens_used=100,
            )
            detector.analyze(step)
        assert detector._baseline_established is True
        assert detector._baseline_velocity is not None


class TestTokenVelocityDetection:
    """Test velocity spike detection."""

    def test_spike_triggers_detection(self):
        """A sudden spike in token usage triggers detection."""
        detector = TokenVelocityDetector(
            warmup=3, velocity_multiplier=3.0, window=3,
        )

        # Establish baseline with moderate tokens
        for i in range(3):
            step = AgentStep(
                step_number=i + 1,
                thought="thinking",
                tokens_used=100,
            )
            detector.analyze(step)

        # Spike to 10x tokens
        step = AgentStep(
            step_number=4,
            thought="thinking with huge context",
            tokens_used=5000,
        )
        result = detector.analyze(step)

        assert result.detected is True
        assert result.pattern == "token_velocity"
        assert "current_velocity" in result.evidence
        assert "baseline_velocity" in result.evidence

    def test_normal_usage_no_detection(self):
        """Normal token usage doesn't trigger detection."""
        detector = TokenVelocityDetector(
            warmup=3, velocity_multiplier=3.0, window=5,
        )

        # Consistent moderate usage
        for i in range(10):
            step = AgentStep(
                step_number=i + 1,
                thought="thinking",
                tokens_used=100 + i * 10,  # Gradual increase
            )
            result = detector.analyze(step)
            assert result.detected is False


class TestTokenVelocityBaseline:
    """Test baseline adaptation."""

    def test_baseline_adapts_ema(self):
        """Baseline slowly adapts with exponential moving average."""
        detector = TokenVelocityDetector(
            warmup=2, velocity_multiplier=5.0, window=3,
        )

        # Establish baseline at 100
        for i in range(2):
            step = AgentStep(
                step_number=i + 1, thought="t", tokens_used=100,
            )
            detector.analyze(step)

        initial_baseline = detector._baseline_velocity

        # Gradual increase — baseline should adapt
        for i in range(10):
            step = AgentStep(
                step_number=i + 3, thought="t", tokens_used=120,
            )
            detector.analyze(step)

        # Baseline should have moved toward 120
        assert detector._baseline_velocity > initial_baseline


class TestTokenVelocityEdgeCases:
    """Test edge cases."""

    def test_zero_tokens(self):
        """Zero tokens per step shouldn't crash."""
        detector = TokenVelocityDetector(warmup=2, velocity_multiplier=3.0)
        for i in range(5):
            step = AgentStep(
                step_number=i + 1, thought="t", tokens_used=0,
            )
            result = detector.analyze(step)
            assert result.detected is False

    def test_name_property(self):
        """Name property returns correct string."""
        detector = TokenVelocityDetector()
        assert detector.name == "TokenVelocityDetector"


class TestTokenVelocityReset:
    """Test the reset method."""

    def test_reset_clears_state(self):
        """After reset, baseline is re-established from scratch."""
        detector = TokenVelocityDetector(warmup=2, velocity_multiplier=3.0)

        # Establish baseline
        for i in range(3):
            step = AgentStep(
                step_number=i + 1, thought="t", tokens_used=100,
            )
            detector.analyze(step)

        detector.reset()

        assert detector._baseline_established is False
        assert detector._baseline_velocity is None
        assert detector._step_count == 0
