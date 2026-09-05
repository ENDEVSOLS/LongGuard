"""ToolRepeatDetector — detects identical tool calls repeated within a sliding window."""

from __future__ import annotations

from collections import deque

from ..step import AgentStep
from .base import AbstractLoopDetector, DetectionResult


class ToolRepeatDetector(AbstractLoopDetector):
    """Detects when the same tool is called with identical arguments multiple times.

    This is the most straightforward loop pattern: the agent calls the same tool
    with the same arguments repeatedly, producing no new information. The detector
    uses a sliding window over recent steps and counts occurrences of the same
    action hash (MD5 of tool name + arguments).

    Importantly, this detector does **not** flag legitimate polling patterns where
    the same tool is called with **different** arguments — those produce different
    hashes and are not counted as repeats.

    Args:
        repeat_threshold: Number of identical calls within the window that triggers
            detection. Default is 3 (same call 3 times = loop).
        window: Size of the sliding window in steps. The detector only looks at
            the most recent ``window`` steps. Default is 6.
    """

    def __init__(self, repeat_threshold: int = 3, window: int = 6) -> None:
        self.repeat_threshold = repeat_threshold
        self.window = window
        self._history: deque[str] = deque(maxlen=window)
        self._tool_names: deque[str] = deque(maxlen=window)

    def analyze(self, step: AgentStep) -> DetectionResult:
        """Analyze a step for tool-repeat pattern.

        If the step has no tool call, it is appended to the window but never
        triggers detection. Only steps with ``action is not None`` are checked.

        Args:
            step: The agent step to analyze.

        Returns:
            DetectionResult with pattern ``"tool_repeat"`` if the same action hash
            appears at least ``repeat_threshold`` times in the sliding window.
        """
        if step.action is None:
            # No tool call — still track the step for window sizing
            self._history.append("")
            self._tool_names.append("")
            return DetectionResult(detected=False)

        action_hash = step.action_hash
        self._history.append(action_hash)
        self._tool_names.append(step.action)

        count = self._history.count(action_hash)

        if count >= self.repeat_threshold:
            confidence = min(count / self.repeat_threshold, 1.0)
            return DetectionResult(
                detected=True,
                pattern="tool_repeat",
                confidence=confidence,
                evidence={
                    "tool": step.action,
                    "repeated_times": count,
                    "hash": action_hash,
                    "window_size": len(self._history),
                },
            )

        return DetectionResult(detected=False)

    def reset(self) -> None:
        """Clear the sliding window and all internal state."""
        self._history.clear()
        self._tool_names.clear()

    @property
    def name(self) -> str:
        return "ToolRepeatDetector"
