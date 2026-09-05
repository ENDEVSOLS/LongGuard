"""Tests for GuardReport."""

import json

from longguard.core.breaker import BreakerState
from longguard.core.detectors.base import DetectionResult
from longguard.core.reporter import DetectionEvent, GuardReport
from longguard.core.step import AgentStep


class TestGuardReportRecording:
    """Test step and detection recording."""

    def test_record_step(self):
        """record_step updates total steps and tokens."""
        report = GuardReport()
        step = AgentStep(
            step_number=1,
            thought="thinking",
            tokens_used=150,
            latency_ms=50.0,
        )
        report.record_step(step)
        assert report.total_steps == 1
        assert report.total_tokens == 150
        assert len(report.step_timeline) == 1

    def test_record_multiple_steps(self):
        """Multiple steps accumulate correctly."""
        report = GuardReport()
        for i in range(5):
            step = AgentStep(
                step_number=i + 1,
                thought="thinking",
                tokens_used=100,
            )
            report.record_step(step)
        assert report.total_steps == 5
        assert report.total_tokens == 500

    def test_record_detection(self):
        """record_detection adds a DetectionEvent."""
        report = GuardReport()
        result = DetectionResult(
            detected=True,
            pattern="tool_repeat",
            confidence=0.9,
            evidence={"tool": "search"},
        )
        report.record_detection(3, result)
        assert len(report.detections) == 1
        assert report.detections[0].step_number == 3
        assert report.detections[0].pattern == "tool_repeat"

    def test_record_reflection(self):
        """record_reflection increments the counter."""
        report = GuardReport()
        assert report.reflections_injected == 0
        report.record_reflection()
        assert report.reflections_injected == 1
        report.record_reflection()
        assert report.reflections_injected == 2


class TestGuardReportFinalize:
    """Test report finalization."""

    def test_finalize_with_kill(self):
        """finalize sets state and kill reason."""
        report = GuardReport()
        report.finalize(BreakerState.OPEN, kill_reason="token_budget_exceeded")
        assert report.final_state == BreakerState.OPEN
        assert report.kill_reason == "token_budget_exceeded"

    def test_finalize_clean_exit(self):
        """finalize for clean exit has no kill reason."""
        report = GuardReport()
        report.finalize(BreakerState.CLOSED)
        assert report.final_state == BreakerState.CLOSED
        assert report.kill_reason is None


class TestGuardReportSerialization:
    """Test report serialization."""

    def test_to_dict(self):
        """to_dict produces a valid dictionary."""
        report = GuardReport()
        step = AgentStep(step_number=1, thought="test", tokens_used=50)
        report.record_step(step)
        result = DetectionResult(
            detected=True, pattern="tool_repeat",
            confidence=0.9, evidence={"tool": "search"},
        )
        report.record_detection(1, result)
        report.record_reflection()
        report.finalize(BreakerState.OPEN, kill_reason="loop")

        d = report.to_dict()
        assert d["total_steps"] == 1
        assert d["total_tokens"] == 50
        assert len(d["detections"]) == 1
        assert d["reflections_injected"] == 1
        assert d["final_state"] == "open"
        assert d["kill_reason"] == "loop"

    def test_to_json(self):
        """to_json produces valid JSON."""
        report = GuardReport()
        step = AgentStep(step_number=1, thought="test", tokens_used=50)
        report.record_step(step)
        report.finalize(BreakerState.CLOSED)

        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert parsed["total_tokens"] == 50

    def test_roundtrip_json(self):
        """JSON serialization roundtrip preserves key data."""
        report = GuardReport()
        for i in range(3):
            step = AgentStep(step_number=i + 1, thought="test", tokens_used=100)
            report.record_step(step)
        report.finalize(BreakerState.CLOSED)

        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert parsed["total_steps"] == 3
        assert parsed["total_tokens"] == 300


class TestGuardReportSummary:
    """Test the human-readable summary."""

    def test_summary_basic(self):
        """Summary includes key information."""
        report = GuardReport()
        step = AgentStep(step_number=1, thought="test", tokens_used=100)
        report.record_step(step)
        report.finalize(BreakerState.CLOSED)

        summary = report.summary()
        assert "Total Steps: 1" in summary
        assert "100" in summary
        assert "closed" in summary

    def test_summary_with_kill(self):
        """Summary includes kill reason when present."""
        report = GuardReport()
        report.finalize(BreakerState.OPEN, kill_reason="token_budget_exceeded")

        summary = report.summary()
        assert "Kill Reason" in summary

    def test_summary_with_detections(self):
        """Summary includes detection details."""
        report = GuardReport()
        result = DetectionResult(
            detected=True, pattern="tool_repeat",
            confidence=0.9, evidence={},
        )
        report.record_detection(5, result)
        report.finalize(BreakerState.REFLECTING)

        summary = report.summary()
        assert "tool_repeat" in summary


class TestDetectionEvent:
    """Test DetectionEvent dataclass."""

    def test_to_dict(self):
        """DetectionEvent serializes to dict."""
        event = DetectionEvent(
            step_number=3,
            pattern="semantic_oscillation",
            confidence=0.8,
            evidence={"thought_variance": 0.05},
        )
        d = event.to_dict()
        assert d["step_number"] == 3
        assert d["pattern"] == "semantic_oscillation"
        assert d["confidence"] == 0.8
