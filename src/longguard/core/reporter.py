"""GuardReport — captures a full LongGuard run summary for observability."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .detectors.base import DetectionResult
from .step import AgentStep

if TYPE_CHECKING:
    from .breaker import BreakerState

logger = logging.getLogger("longguard.reporter")


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DetectionEvent:
        return cls(
            step_number=data["step_number"],
            pattern=data["pattern"],
            confidence=data["confidence"],
            evidence=data.get("evidence", {}),
        )


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
    - Estimated dollar cost (when model pricing is available)

    Attributes:
        total_steps: Total number of agent steps in the run.
        total_tokens: Total tokens consumed across all steps.
        detections: List of DetectionEvent instances.
        reflections_injected: Number of Reflect & Pivot prompts injected.
        final_state: The CircuitBreaker state at run end.
        kill_reason: Why the agent was killed (None if clean exit).
        step_timeline: Per-step token usage and timing.
        estimated_cost_usd: Estimated dollar cost of the run.  ``None`` when
            model pricing cannot be determined.
        model: Model identifier used for cost estimation (informational).
    """

    total_steps: int = 0
    total_tokens: int = 0
    detections: list[DetectionEvent] = field(default_factory=list)
    reflections_injected: int = 0
    final_state: BreakerState | None = None
    kill_reason: str | None = None
    step_timeline: list[dict[str, Any]] = field(default_factory=list)
    estimated_cost_usd: float | None = None
    model: str | None = None

    def record_step(
        self,
        step: AgentStep,
        total_tokens: int | None = None,
        total_steps: int | None = None,
        cost_delta: float | None = None,
    ) -> None:
        """Record a step in the timeline.

        Args:
            step: The agent step to record.
            total_tokens: Authoritative total token count from the breaker.
                If provided, overrides the report's running total.
            total_steps: Authoritative total step count from the breaker.
                If provided, overrides the report's running total.
            cost_delta: Additional USD cost to add for this step.  ``None``
                means no cost information is available for this step.
        """
        if total_steps is not None:
            self.total_steps = total_steps
        else:
            self.total_steps = max(self.total_steps, step.step_number)

        if total_tokens is not None:
            self.total_tokens = total_tokens
        else:
            self.total_tokens += step.tokens_used

        if cost_delta is not None:
            if self.estimated_cost_usd is None:
                self.estimated_cost_usd = cost_delta
            else:
                self.estimated_cost_usd += cost_delta

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
        if self.estimated_cost_usd is not None:
            model_label = f" ({self.model})" if self.model else ""
            lines.append(f"Estimated Cost: ${self.estimated_cost_usd:.4f} USD{model_label}")
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
            "estimated_cost_usd": self.estimated_cost_usd,
            "model": self.model,
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

    # ------------------------------------------------------------------
    # Feature 3: save / load
    # ------------------------------------------------------------------

    def save(self, path: str | Path, *, fmt: str | None = None) -> None:
        """Save the report to a file on disk.

        The format is inferred from the file extension unless overridden via
        the ``fmt`` argument.  Supported formats:

        * ``"json"`` (default, always available)
        * ``"yaml"`` — requires ``PyYAML`` (``pip install pyyaml``).  Falls
          back to JSON with a ``".json"`` extension if PyYAML is not installed.

        Args:
            path: Destination file path.  Parent directories must exist.
            fmt: Override format — ``"json"`` or ``"yaml"``.  When ``None``
                the format is inferred from the file extension
                (``.json``, ``.yaml``, or ``.yml``).

        Example::

            report.save("run_report.json")
            report.save("run_report.yaml")
        """
        path = Path(path)

        # Determine format
        if fmt is None:
            suffix = path.suffix.lower()
            if suffix in (".yaml", ".yml"):
                fmt = "yaml"
            else:
                fmt = "json"

        data = self.to_dict()

        if fmt == "yaml":
            try:
                import yaml  # type: ignore[import-untyped]
                content = yaml.dump(data, allow_unicode=True, sort_keys=False)
                path.write_text(content, encoding="utf-8")
                return
            except ImportError:
                logger.warning(
                    "PyYAML is not installed; saving as JSON instead.  "
                    "Install it with: pip install pyyaml"
                )
                path = path.with_suffix(".json")

        # JSON (default / fallback)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> GuardReport:
        """Load a previously saved report from disk.

        Supports both JSON and YAML files (YAML requires ``PyYAML``).

        Args:
            path: Path to the saved report file.

        Returns:
            A fully reconstructed :class:`GuardReport` instance.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the file format cannot be determined or the content
                is invalid.

        Example::

            report = GuardReport.load("run_report.json")
            print(report.summary())
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Report file not found: {path}")

        suffix = path.suffix.lower()
        content = path.read_text(encoding="utf-8")

        if suffix in (".yaml", ".yml"):
            try:
                import yaml
                data: dict[str, Any] = yaml.safe_load(content)
            except ImportError as exc:
                raise ImportError(
                    "PyYAML is required to load YAML reports.  "
                    "Install it with: pip install pyyaml"
                ) from exc
        elif suffix == ".json":
            data = json.loads(content)
        else:
            # Attempt JSON regardless
            try:
                data = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Cannot determine format for {path}.  "
                    "Use a .json or .yaml/.yml extension."
                ) from exc

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> GuardReport:
        """Reconstruct a GuardReport from a dictionary (produced by to_dict)."""
        from .breaker import BreakerState  # local import avoids circular

        final_state: BreakerState | None = None
        raw_state = data.get("final_state")
        if raw_state is not None:
            try:
                final_state = BreakerState(raw_state)
            except ValueError:
                logger.warning("Unknown BreakerState value %r — skipping.", raw_state)

        report = cls(
            total_steps=data.get("total_steps", 0),
            total_tokens=data.get("total_tokens", 0),
            reflections_injected=data.get("reflections_injected", 0),
            final_state=final_state,
            kill_reason=data.get("kill_reason"),
            step_timeline=data.get("step_timeline", []),
            estimated_cost_usd=data.get("estimated_cost_usd"),
            model=data.get("model"),
        )
        for det_data in data.get("detections", []):
            report.detections.append(DetectionEvent.from_dict(det_data))
        return report
