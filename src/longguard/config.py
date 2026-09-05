"""LongGuard configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GuardConfig:
    """Configuration for LongGuard circuit breaker.

    All parameters have sensible defaults. Override only what you need.

    Attributes:
        tool_repeat_threshold: Number of identical tool calls that triggers detection.
        tool_repeat_window: Sliding window size (in steps) for tool repeat detection.
        semantic_variance_threshold: Variance below which thoughts are considered oscillating.
        semantic_window: Number of recent thoughts to analyze for oscillation.
        dead_end_threshold: Consecutive no-progress steps before dead-end detection.
        dead_end_progress_threshold: Similarity threshold below which observations
            are considered to have made progress (lower = more different = progress).
            When ``sentence-transformers`` is installed, cosine similarity on
            embeddings is used. Otherwise, Jaccard similarity is used as fallback.
        token_velocity_multiplier: Multiplier over baseline velocity that triggers a spike.
        token_velocity_window: Sliding window for computing rolling token velocity.
        token_velocity_warmup: Number of steps before velocity baseline is established.
        max_tokens_per_run: Hard token budget cap per agent run.
        max_steps: Hard step cap per agent run.
        max_reflections: Number of reflection attempts before hard kill.
        pivot_templates: Custom pivot prompt templates. Keys must be one of:
            "tool_repeat", "semantic_oscillation", "dead_end_drift", "token_velocity".
            Values are strings with template variables in {curly_braces}.
        log_level: Logging level for LongGuard internals.
        emit_events: Whether to emit events for LangSmith / custom callbacks.
    """

    # Loop detection thresholds
    tool_repeat_threshold: int = 3
    tool_repeat_window: int = 6
    semantic_variance_threshold: float = 0.15
    semantic_window: int = 8
    dead_end_threshold: int = 5
    dead_end_progress_threshold: float = 0.6
    token_velocity_multiplier: float = 3.0
    token_velocity_window: int = 5
    token_velocity_warmup: int = 3

    # Budget caps
    max_tokens_per_run: int = 50_000
    max_steps: int = 30

    # Reflection settings
    max_reflections: int = 2
    pivot_templates: dict[str, str] = field(default_factory=dict)

    # Monitoring
    log_level: str = "WARNING"
    emit_events: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.tool_repeat_threshold < 1:
            raise ValueError("tool_repeat_threshold must be >= 1")
        if self.tool_repeat_window < self.tool_repeat_threshold:
            raise ValueError("tool_repeat_window must be >= tool_repeat_threshold")
        if self.semantic_variance_threshold <= 0:
            raise ValueError("semantic_variance_threshold must be > 0")
        if self.semantic_window < 2:
            raise ValueError("semantic_window must be >= 2")
        if self.dead_end_threshold < 1:
            raise ValueError("dead_end_threshold must be >= 1")
        if self.token_velocity_multiplier <= 1.0:
            raise ValueError("token_velocity_multiplier must be > 1.0")
        if self.token_velocity_window < 2:
            raise ValueError("token_velocity_window must be >= 2")
        if self.max_tokens_per_run < 1:
            raise ValueError("max_tokens_per_run must be >= 1")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.max_reflections < 1:
            raise ValueError("max_reflections must be >= 1")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GuardConfig:
        """Create a GuardConfig from a dictionary.

        Unknown keys are silently ignored so that config files can contain
        extra metadata without breaking parsing.

        Args:
            data: Dictionary of configuration values.

        Returns:
            A GuardConfig instance.

        Example::

            config = GuardConfig.from_dict({
                "tool_repeat_threshold": 5,
                "max_tokens_per_run": 100_000,
            })
        """
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the config to a dictionary."""
        import dataclasses

        return dataclasses.asdict(self)

    def merge(self, other: GuardConfig | None = None, **kwargs: Any) -> GuardConfig:
        """Create a new config by merging with another config or keyword overrides.

        Args:
            other: Another GuardConfig whose non-default values take precedence.
            **kwargs: Individual field overrides (highest precedence).

        Returns:
            A new GuardConfig with merged values.
        """
        base = self.to_dict()
        if other is not None:
            other_dict = other.to_dict()
            defaults = GuardConfig()
            default_dict = defaults.to_dict()
            for key, value in other_dict.items():
                if value != default_dict[key]:
                    base[key] = value
        base.update(kwargs)
        return GuardConfig.from_dict(base)
