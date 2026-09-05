"""TokenVelocityDetector — detects cost spikes from runaway token consumption."""

from __future__ import annotations

from collections import deque

import numpy as np

from ..step import AgentStep
from .base import AbstractLoopDetector, DetectionResult


class TokenVelocityDetector(AbstractLoopDetector):
    """Detects when token consumption velocity spikes, indicating runaway cost.

    In normal operation, an agent consumes tokens at a roughly steady rate per
    step. A sudden velocity spike — e.g., a single step consuming 10x the
    normal tokens — often indicates the agent is stuck in a loop where each
    iteration burns more tokens (e.g., accumulating context in a growing
    conversation).

    This detector computes a rolling average of tokens consumed per step over
    a sliding window. After a warmup period, it establishes a baseline velocity.
    If the current velocity exceeds the baseline by ``velocity_multiplier``,
    it triggers detection.

    Additionally, the detector tracks the second derivative (acceleration) of
    token consumption. A positive acceleration indicates the rate is increasing,
    which is a strong signal of a cost runaway.

    Args:
        velocity_multiplier: How many times over the baseline velocity triggers
            detection. Default is 3.0 (3x the baseline = spike).
        window: Size of the sliding window for computing rolling velocity.
            Default is 5.
        warmup: Number of steps before the baseline is established. Before
            warmup, the detector never triggers. Default is 3.
    """

    def __init__(
        self,
        velocity_multiplier: float = 3.0,
        window: int = 5,
        warmup: int = 3,
    ) -> None:
        self.velocity_multiplier = velocity_multiplier
        self.window = window
        self.warmup = warmup
        self._token_history: deque[int] = deque(maxlen=window)
        self._step_count: int = 0
        self._baseline_velocity: float | None = None
        self._baseline_established: bool = False
        self._velocities: deque[float] = deque(maxlen=window * 2)

    def _compute_velocity(self) -> float:
        """Compute the current rolling average token velocity.

        Returns:
            Average tokens per step over the sliding window.
        """
        if not self._token_history:
            return 0.0
        return float(np.mean(list(self._token_history)))

    def _compute_acceleration(self) -> float:
        """Compute the acceleration (second derivative) of token consumption.

        Positive acceleration means velocity is increasing.

        Returns:
            The acceleration value. 0.0 if insufficient data.
        """
        if len(self._velocities) < 2:
            return 0.0
        recent = list(self._velocities)[-min(len(self._velocities), self.window) :]
        if len(recent) < 2:
            return 0.0
        # Simple second difference
        diffs = np.diff(recent)
        return float(np.mean(diffs))

    def analyze(self, step: AgentStep) -> DetectionResult:
        """Analyze a step for token velocity spikes.

        During the warmup period, the detector collects data but never triggers.
        After warmup, the first velocity reading becomes the baseline. Subsequent
        velocities are compared against this baseline multiplied by
        ``velocity_multiplier``.

        The baseline is updated lazily: if the current velocity is below the
        threshold, it's incorporated into the baseline (EMA with alpha=0.3).
        This allows the baseline to drift downward for efficient agents while
        still catching spikes.

        Args:
            step: The agent step to analyze.

        Returns:
            DetectionResult with pattern ``"token_velocity"`` if the current
            velocity exceeds the baseline by ``velocity_multiplier``.
        """
        self._step_count += 1
        self._token_history.append(step.tokens_used)

        current_velocity = self._compute_velocity()
        self._velocities.append(current_velocity)

        # During warmup, establish baseline but don't trigger
        if self._step_count <= self.warmup:
            if self._step_count == self.warmup:
                self._baseline_velocity = current_velocity
                self._baseline_established = True
            return DetectionResult(detected=False)

        if not self._baseline_established or self._baseline_velocity is None:
            self._baseline_velocity = current_velocity
            self._baseline_established = True
            return DetectionResult(detected=False)

        # Avoid division by zero for zero-baseline
        if self._baseline_velocity == 0.0:
            # If baseline is zero but we're consuming tokens now, update baseline
            if current_velocity > 0:
                self._baseline_velocity = current_velocity
            return DetectionResult(detected=False)

        # Check for velocity spike
        ratio = current_velocity / self._baseline_velocity

        if ratio >= self.velocity_multiplier:
            acceleration = self._compute_acceleration()
            return DetectionResult(
                detected=True,
                pattern="token_velocity",
                confidence=min(ratio / self.velocity_multiplier, 1.0),
                evidence={
                    "current_velocity": current_velocity,
                    "baseline_velocity": self._baseline_velocity,
                    "velocity_ratio": ratio,
                    "acceleration": acceleration,
                    "step_number": step.step_number,
                    "tokens_this_step": step.tokens_used,
                },
            )

        # Update baseline with EMA (exponential moving average)
        # Only update if velocity is normal — don't let spikes inflate the baseline
        alpha = 0.3
        self._baseline_velocity = (
            alpha * current_velocity + (1 - alpha) * self._baseline_velocity
        )

        return DetectionResult(detected=False)

    def reset(self) -> None:
        """Clear all tracking state and baseline."""
        self._token_history.clear()
        self._step_count = 0
        self._baseline_velocity = None
        self._baseline_established = False
        self._velocities.clear()

    @property
    def name(self) -> str:
        return "TokenVelocityDetector"
