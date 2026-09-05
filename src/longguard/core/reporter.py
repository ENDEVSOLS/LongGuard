"""GuardReport — captures a full LongGuard run summary for observability."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .detectors.base import DetectionResult
from .step import AgentStep

if TYPE_CHECKING:
    from .breaker import BreakerState


@dataclass
class DetectionEvent:
    """A single detection event recorded during a run.

    Attributes:
        step_number: The step at which detection occurred.
        pattern: The detected pattern name.
        confidence: Detection confidence.
        evidence: Detector-specific evidence.
    """

    step_number: int
    pattern: str
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_number": self.step_number,
            "pattern": self.pattern,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


@dataclass
class GuardReport:
    """Full run summary produced by LongGuard.

    The report is updated incrementally as the agent runs and is finalized
    when the run ends (either naturally or via kill). It captures:

    - Total steps and tokens consumed
    - All loop detections that occurred
    - Number of Reflect & Pivot interventions injected
    - The final breaker state (CLOSED = clean, OPEN = killed)
    - The kill reason (if the agent was terminated)

    Attributes:
        total_steps: Total number of agent steps in the run.
        total_tokens: Total tokens consumed across all steps.
        detections: List of DetectionEvent instances.
        reflections_injected: Number of Reflect & Pivot prompts injected.
        final_state: The CircuitBreaker state at run end.
        kill_reason: Why the agent was killed (None if clean exit).
        step_timeline: Per-step token usage and timing.
    """

    total_steps: int = 0
    total_tokens: int = 0
    detections: list[DetectionEvent] = field(default_factory=list)
    reflections_injected: int = 0
    final_state: BreakerState | None = None
    kill_reason: str | None = None
    step_timeline: list[dict[str, Any]] = field(default_factory=list)

    def record_step(
        self,
        step: AgentStep,
        total_tokens: int | None = None,
        total_steps: int | None = None,
    ) -> None:
        """Record a step in the timeline.

        Args:
            step: The agent step to record.
            total_tokens: Authoritative total token count from the breaker.
                If provided, overrides the report's running total.
            total_steps: Authoritative total step count from the breaker.
                If provided, overrides the report's running total.
        """
        if total_steps is not None:
            self.total_steps = total_steps
        else:
            self.total_steps = max(self.total_steps, step.step_number)

        if total_tokens is not None:
            self.total_tokens = total_tokens
        else:
            self.total_tokens += step.tokens_used

        self.step_timeline.append({
            "step_number": step.step_number,
            "tokens_used": step.tokens_used,
            "latency_ms": step.latency_ms,
            "action": step.action,
        })

    def record_detection(self, step_number: int, result: DetectionResult) -> None:
        """Record a detection event.

        Args:
            step_number: The step at which detection occurred.
            result: The DetectionResult from the detector.
        """
        self.detections.append(DetectionEvent(
            step_number=step_number,
            pattern=result.pattern,
            confidence=result.confidence,
            evidence=result.evidence,
        ))

    def record_reflection(self) -> None:
        """Record that a Reflect & Pivot prompt was injected."""
        self.reflections_injected += 1

    def finalize(
        self,
        final_state: BreakerState,
        kill_reason: str | None = None,
    ) -> None:
        """Finalize the report at the end of a run.

        Args:
            final_state: The CircuitBreaker's final state.
            kill_reason: Why the agent was killed, if applicable.
        """
        self.final_state = final_state
        self.kill_reason = kill_reason

    def summary(self) -> str:
        """Generate a human-readable summary of the report.

        Returns:
            A formatted multi-line string summarizing the run.
        """
        lines = [
            "=== LongGuard Run Report ===",
            f"Total Steps: {self.total_steps}",
            f"Total Tokens: {self.total_tokens:,}",
            f"Final State: {self.final_state.value if self.final_state else 'unknown'}",
            f"Detections: {len(self.detections)}",
            f"Reflections Injected: {self.reflections_injected}",
        ]
        if self.kill_reason:
            lines.append(f"Kill Reason: {self.kill_reason}")
        if self.detections:
            lines.append("")
            lines.append("Detection Details:")
            for det in self.detections:
                lines.append(
                    f"  Step {det.step_number}: {det.pattern} "
                    f"(confidence: {det.confidence:.0%})"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to a dictionary.

        Returns:
            A dictionary representation of the report.
        """
        return {
            "total_steps": self.total_steps,
            "total_tokens": self.total_tokens,
            "detections": [d.to_dict() for d in self.detections],
            "reflections_injected": self.reflections_injected,
            "final_state": self.final_state.value if self.final_state else None,
            "kill_reason": self.kill_reason,
            "step_timeline": self.step_timeline,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the report to JSON.

        Args:
            indent: JSON indentation level.

        Returns:
            A JSON string.
        """
        return json.dumps(self.to_dict(), indent=indent)
