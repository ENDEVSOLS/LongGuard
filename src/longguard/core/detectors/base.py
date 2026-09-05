"""Abstract base class for all loop detectors and the DetectionResult dataclass."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..step import AgentStep


@dataclass
class DetectionResult:
    """Result returned by every detector's ``analyze()`` method.

    Attributes:
        detected: Whether a loop pattern was detected.
        pattern: Name of the detected pattern (e.g. ``"tool_repeat"``).
            Empty string if not detected.
        confidence: Confidence score in [0.0, 1.0]. Higher means more certain.
            Zero if not detected.
        evidence: Dictionary of detector-specific evidence for observability
            and for filling Reflect & Pivot templates.
    """

    detected: bool = False
    pattern: str = ""
    confidence: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the detection result."""
        if self.detected:
            if not self.pattern:
                raise ValueError("pattern must be set when detected=True")
            if not (0.0 <= self.confidence <= 1.0):
                raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")

    def merge(self, other: DetectionResult) -> DetectionResult:
        """Merge two detection results, preferring the one with higher confidence.

        If both detected, returns the one with higher confidence. If only one
        detected, returns that one. If neither detected, returns a clean result.

        Args:
            other: Another DetectionResult to merge with.

        Returns:
            The merged DetectionResult.
        """
        if not self.detected and not other.detected:
            return DetectionResult(detected=False)
        if self.detected and not other.detected:
            return self
        if not self.detected and other.detected:
            return other
        # Both detected — keep higher confidence
        if self.confidence >= other.confidence:
            return self
        return other

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "detected": self.detected,
            "pattern": self.pattern,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


class AbstractLoopDetector(ABC):
    """Abstract base class for all loop pattern detectors.

    Every detector must implement ``analyze()`` which is called once per
    agent step. The detector maintains internal state (sliding windows,
    history buffers) and resets between runs via ``reset()``.
    """

    @abstractmethod
    def analyze(self, step: AgentStep) -> DetectionResult:
        """Analyze a single agent step for a loop pattern.

        Called by the CircuitBreaker after every agent step. The detector
        may update internal state and return a DetectionResult indicating
        whether a pattern was found.

        Args:
            step: The agent step to analyze.

        Returns:
            A DetectionResult indicating detection status, pattern name,
            confidence, and evidence.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the detector's internal state.

        Called between agent runs to clear sliding windows, history buffers,
        and any accumulated state. Must not raise.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this detector (e.g. ``"ToolRepeatDetector"``)."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
