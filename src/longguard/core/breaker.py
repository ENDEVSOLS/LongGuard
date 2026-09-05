"""CircuitBreaker state machine — the core decision engine of LongGuard."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from enum import Enum
from typing import Any

from ..config import GuardConfig
from .detectors.base import AbstractLoopDetector, DetectionResult
from .detectors.dead_end import DeadEndDriftDetector
from .detectors.semantic_osc import Embedder, SemanticOscillationDetector
from .detectors.token_velocity import TokenVelocityDetector
from .detectors.tool_repeat import ToolRepeatDetector
from .pivot import ReflectAndPivotInjector
from .reporter import GuardReport
from .step import AgentStep

logger = logging.getLogger("longguard")


class BreakerState(Enum):
    """States of the circuit breaker.

    The breaker follows a state machine:

        CLOSED ──(detection)──▶ REFLECTING ──(detection)──▶ HALF_OPEN ──(detection)──▶ OPEN
            ▲                       │                           │
            │                       │                           │
            └───────────────────────┘                           │
              (clean step after                                   │
               reflection)                                        │
                                                                KILL
    - CLOSED: Normal operation. No loops detected.
    - REFLECTING: A loop was detected. A Reflect & Pivot prompt is injected.
    - HALF_OPEN: After reflection, the agent gets one more chance. If the
      pattern persists, the breaker opens.
    - OPEN: The agent is terminated. No further steps are allowed.
    """

    CLOSED = "closed"
    REFLECTING = "reflecting"
    HALF_OPEN = "half_open"
    OPEN = "open"


class BreakerDecision:
    """Decision returned by the CircuitBreaker after each step.

    Attributes:
        action: One of ``"continue"``, ``"reflect"``, or ``"kill"``.
        reason: Human-readable reason for the decision (empty for continue).
        inject_prompt: The Reflect & Pivot prompt to inject (only for ``"reflect"``).
        report: The current GuardReport snapshot.
    """

    def __init__(
        self,
        action: str,
        reason: str = "",
        inject_prompt: str | None = None,
        report: GuardReport | None = None,
    ) -> None:
        if action not in ("continue", "reflect", "kill"):
            raise ValueError(f"Invalid breaker action: {action!r}")
        self.action = action
        self.reason = reason
        self.inject_prompt = inject_prompt
        self.report = report

    def __repr__(self) -> str:
        parts = [f"BreakerDecision(action={self.action!r}"]
        if self.reason:
            parts.append(f", reason={self.reason!r}")
        if self.inject_prompt:
            parts.append(", has_pivot_prompt=True")
        return ", ".join(parts) + ")"

    def __bool__(self) -> bool:
        """Truthiness — True unless the action is 'kill'."""
        return self.action != "kill"


# Type for event callbacks
EventCallback = Callable[[str, dict[str, Any]], None]


class CircuitBreaker:
    """The core decision engine that orchestrates loop detection and intervention.

    The CircuitBreaker is called after every agent step. It runs all registered
    detectors, evaluates the combined results against its state machine, and
    returns a BreakerDecision telling the integration layer what to do:

    - ``continue``: No issues detected. Proceed normally.
    - ``reflect``: A loop was detected. Inject the provided pivot prompt.
    - ``kill``: The agent is stuck beyond recovery. Terminate gracefully.

    The breaker also enforces hard caps on token budget and step count,
    regardless of detector results.

    Args:
        config: GuardConfig with all tuning parameters.
        event_callback: Optional callback for emitting events to LangSmith
            or custom monitoring systems. Called with (event_name, event_data).
    """

    def __init__(
        self,
        config: GuardConfig | None = None,
        event_callback: EventCallback | None = None,
    ) -> None:
        self.config = config or GuardConfig()
        self.state = BreakerState.CLOSED
        self._event_callback = event_callback
        self._reflection_count = 0
        self._total_tokens = 0
        self._total_steps = 0
        self._report = GuardReport()
        self._lock = threading.Lock()

        # Event debouncing: minimum interval between events of the same type
        self._event_debounce_ms = 100
        self._last_event_times: dict[str, float] = {}

        # Initialize detectors from config
        self.detectors: list[AbstractLoopDetector] = self._init_detectors()

        # Initialize pivot injector
        self.pivot = ReflectAndPivotInjector(self.config.pivot_templates or None)

    def _init_detectors(self) -> list[AbstractLoopDetector]:
        """Create detector instances based on config.

        Creates a shared embedder for both the SemanticOscillationDetector
        and DeadEndDriftDetector. If ``sentence-transformers`` is installed
        (via ``pip install longguard[embeddings]``), both detectors get a
        high-quality ``SentenceTransformerEmbedder``. Otherwise,
        SemanticOscillationDetector falls back to ``HashBasedEmbedder``
        and DeadEndDriftDetector falls back to Jaccard similarity.

        Returns:
            List of initialized AbstractLoopDetector instances.
        """
        shared_embedder = self._create_shared_embedder()

        return [
            ToolRepeatDetector(
                repeat_threshold=self.config.tool_repeat_threshold,
                window=self.config.tool_repeat_window,
            ),
            SemanticOscillationDetector(
                window=self.config.semantic_window,
                variance_threshold=self.config.semantic_variance_threshold,
                embedder=shared_embedder,
            ),
            DeadEndDriftDetector(
                dead_end_threshold=self.config.dead_end_threshold,
                progress_threshold=self.config.dead_end_progress_threshold,
                embedder=shared_embedder,
            ),
            TokenVelocityDetector(
                velocity_multiplier=self.config.token_velocity_multiplier,
                window=self.config.token_velocity_window,
                warmup=self.config.token_velocity_warmup,
            ),
        ]

    @staticmethod
    def _create_shared_embedder() -> Embedder | None:
        """Create the best available embedder for shared use across detectors.

        Tries to load ``SentenceTransformerEmbedder`` first (requires
        ``pip install longguard[embeddings]``). Falls back to ``None``
        which lets each detector use its own default — ``HashBasedEmbedder``
        for SemanticOscillationDetector, Jaccard for DeadEndDriftDetector.

        Returns:
            A ``SentenceTransformerEmbedder`` if available, else ``None``.
        """
        try:
            from .detectors.semantic_osc import SentenceTransformerEmbedder

            return SentenceTransformerEmbedder()
        except ImportError:
            logger.info(
                "sentence-transformers not installed. Using default embedders. "
                "For best accuracy, install with: pip install longguard[embeddings]"
            )
            return None

    @property
    def report(self) -> GuardReport:
        """The current GuardReport for this run."""
        return self._report

    def check(self, step: AgentStep) -> BreakerDecision:
        """Evaluate an agent step and return a decision.

        This is the main entry point called by integration layers after
        every agent step. Thread-safe via internal lock. The method:

        1. Updates internal counters (tokens, steps)
        2. Checks hard caps (token budget, step count)
        3. Runs all detectors
        4. Applies state machine logic to determine the decision
        5. Emits events for monitoring

        Args:
            step: The AgentStep to evaluate.

        Returns:
            A BreakerDecision indicating what action to take.
        """
        with self._lock:
            return self._check_unlocked(step)

    def _check_unlocked(self, step: AgentStep) -> BreakerDecision:
        """Internal check logic — must be called with self._lock held."""
        # Record the step in the report
        self._total_tokens += step.tokens_used
        self._total_steps += 1
        self._report.record_step(step, self._total_tokens, self._total_steps)

        # === Hard caps ===

        # Token budget exceeded
        if self._total_tokens > self.config.max_tokens_per_run:
            reason = (
                f"token_budget_exceeded: {self._total_tokens:,} > "
                f"{self.config.max_tokens_per_run:,}"
            )
            self.state = BreakerState.OPEN
            self._report.finalize(self.state, kill_reason=reason)
            self._emit_event("kill", {"reason": reason, "total_tokens": self._total_tokens})
            return BreakerDecision(
                action="kill",
                reason=reason,
                report=self._report,
            )

        # Step count exceeded
        if self._total_steps > self.config.max_steps:
            reason = (
                f"max_steps_exceeded: {self._total_steps} > {self.config.max_steps}"
            )
            self.state = BreakerState.OPEN
            self._report.finalize(self.state, kill_reason=reason)
            self._emit_event("kill", {"reason": reason, "total_steps": self._total_steps})
            return BreakerDecision(
                action="kill",
                reason=reason,
                report=self._report,
            )

        # === Run all detectors ===
        best_detection = DetectionResult(detected=False)

        for detector in self.detectors:
            try:
                result = detector.analyze(step)
                if result.detected:
                    self._report.record_detection(step.step_number, result)
                    self._emit_event("detection", {
                        "detector": detector.name,
                        "pattern": result.pattern,
                        "confidence": result.confidence,
                        "evidence": result.evidence,
                    })
                    # Keep the highest-confidence detection
                    best_detection = best_detection.merge(result)
            except Exception as exc:
                logger.warning(
                    "Detector %s raised an exception: %s", detector.name, exc
                )

        # === State machine ===
        if best_detection.detected:
            return self._handle_detection(best_detection, step)

        # If we were in REFLECTING or HALF_OPEN and passed cleanly, recover to CLOSED
        if self.state in (BreakerState.REFLECTING, BreakerState.HALF_OPEN):
            previous_state = self.state
            self.state = BreakerState.CLOSED
            self._emit_event("recovered", {
                "previous_state": previous_state.value,
                "new_state": self.state.value,
            })
            logger.info(
                "Circuit breaker recovered: %s → CLOSED", previous_state.value
            )

        return BreakerDecision(action="continue", report=self._report)

    def _handle_detection(
        self, result: DetectionResult, step: AgentStep
    ) -> BreakerDecision:
        """Handle a positive detection according to the state machine.

        Args:
            result: The DetectionResult that triggered this handler.
            step: The AgentStep where detection occurred.

        Returns:
            A BreakerDecision (reflect or kill).
        """
        if self.state == BreakerState.CLOSED:
            # First detection — inject Reflect & Pivot
            self.state = BreakerState.REFLECTING
            self._reflection_count += 1
            self._report.record_reflection()
            pivot_prompt = self.pivot.generate(result, step)

            self._emit_event("reflect", {
                "pattern": result.pattern,
                "confidence": result.confidence,
                "reflection_count": self._reflection_count,
            })

            logger.info(
                "Loop detected (%s, confidence=%.0f%%). Injecting Reflect & Pivot (#%d).",
                result.pattern,
                result.confidence * 100,
                self._reflection_count,
            )

            return BreakerDecision(
                action="reflect",
                reason=result.pattern,
                inject_prompt=pivot_prompt,
                report=self._report,
            )

        elif self.state == BreakerState.REFLECTING:
            # Already injected reflection — give one more chance (HALF_OPEN)
            # unless we've exhausted reflection attempts
            if self._reflection_count >= self.config.max_reflections:
                return self._kill(result, step)

            self.state = BreakerState.HALF_OPEN
            self._reflection_count += 1
            self._report.record_reflection()
            pivot_prompt = self.pivot.generate(result, step)

            self._emit_event("reflect", {
                "pattern": result.pattern,
                "confidence": result.confidence,
                "reflection_count": self._reflection_count,
                "state": "half_open",
            })

            logger.warning(
                "Loop persists after reflection (%s). "
                "Injecting final Reflect & Pivot (#%d), then HALF_OPEN.",
                result.pattern,
                self._reflection_count,
            )

            return BreakerDecision(
                action="reflect",
                reason=f"reflection_{self._reflection_count}: {result.pattern}",
                inject_prompt=pivot_prompt,
                report=self._report,
            )

        elif self.state == BreakerState.HALF_OPEN:
            # Pattern persists after reflection — hard kill
            return self._kill(result, step)

        # If state is OPEN, already killed
        return BreakerDecision(
            action="kill",
            reason="breaker_already_open",
            report=self._report,
        )

    def _kill(self, result: DetectionResult, step: AgentStep) -> BreakerDecision:
        """Transition to OPEN state and return a kill decision.

        Args:
            result: The DetectionResult that triggered the kill.
            step: The AgentStep where the kill occurred.

        Returns:
            A BreakerDecision with action="kill".
        """
        reason = f"reflection_failed: {result.pattern} (confidence: {result.confidence:.0%})"
        self.state = BreakerState.OPEN
        self._report.finalize(self.state, kill_reason=reason)

        self._emit_event("kill", {
            "reason": reason,
            "pattern": result.pattern,
            "confidence": result.confidence,
            "total_tokens": self._total_tokens,
            "total_steps": self._total_steps,
        })

        logger.error(
            "Circuit breaker OPENED. Agent killed. Reason: %s. "
            "Total tokens: %s, Total steps: %d",
            reason,
            f"{self._total_tokens:,}",
            self._total_steps,
        )

        return BreakerDecision(
            action="kill",
            reason=reason,
            report=self._report,
        )

    def _emit_event(self, event_name: str, data: dict[str, Any]) -> None:
        """Emit an event to the callback, if configured.

        Includes debouncing to prevent flooding monitoring systems.

        Args:
            event_name: Name of the event (e.g. "detection", "reflect", "kill").
            data: Event-specific data dictionary.
        """
        if self._event_callback and self.config.emit_events:
            # Debounce: skip if same event type fired too recently
            now = time.monotonic()
            last = self._last_event_times.get(event_name, 0.0)
            if (now - last) * 1000 < self._event_debounce_ms:
                # "kill" events are never debounced — they're too important
                if event_name != "kill":
                    return
            self._last_event_times[event_name] = now

            try:
                self._event_callback(event_name, data)
            except Exception as exc:
                logger.warning("Event callback raised: %s", exc)

    def reset(self) -> None:
        """Reset the breaker for a new agent run.

        Clears all internal state, resets the state machine to CLOSED,
        and creates a fresh GuardReport. Thread-safe.
        """
        with self._lock:
            self.state = BreakerState.CLOSED
            self._reflection_count = 0
            self._total_tokens = 0
            self._total_steps = 0
            self._report = GuardReport()
            self._last_event_times.clear()
            for detector in self.detectors:
                detector.reset()

    def add_detector(self, detector: AbstractLoopDetector) -> None:
        """Register a custom detector at runtime.

        Custom detectors are appended after the built-in ones and participate
        in the same detection pipeline.

        Args:
            detector: An AbstractLoopDetector instance.
        """
        self.detectors.append(detector)

    @property
    def is_open(self) -> bool:
        """Whether the breaker is in the OPEN (killed) state."""
        return self.state == BreakerState.OPEN

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed in the current run."""
        return self._total_tokens

    @property
    def total_steps(self) -> int:
        """Total steps in the current run."""
        return self._total_steps
