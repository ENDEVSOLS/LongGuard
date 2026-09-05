"""AgentStep dataclass — the fundamental unit of observation in LongGuard."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentStep:
    """Represents a single step in an agent's execution trajectory.

    LongGuard observes every agent step through this data structure. Each detector
    analyzes different aspects of the step to identify loop patterns.

    Attributes:
        step_number: Sequential step index (1-based).
        thought: The agent's chain-of-thought reasoning text for this step.
        action: The tool name the agent decided to call (None if no tool call).
        action_input: The arguments passed to the tool (None if no tool call).
        observation: The result returned by the tool (None if no tool call).
        tokens_used: Number of tokens consumed in this step.
        latency_ms: Wall-clock time for this step in milliseconds.
    """

    step_number: int
    thought: str
    action: str | None = None
    action_input: Any | None = None
    observation: str | None = None
    tokens_used: int = 0
    latency_ms: float = 0.0

    # Private: cached embedding vector (set by detectors, not serialized)
    _thought_vector: list[float] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def action_hash(self) -> str:
        """Fingerprint for ToolRepeatDetector.

        Computes an MD5 hash of the JSON-serialized action + action_input.
        Two steps with the same tool name and same arguments produce the same hash,
        regardless of thought text or observation.

        Returns:
            A hex-encoded MD5 digest string.
        """
        payload = json.dumps(
            {"action": self.action, "input": self.action_input},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def thought_vector(self) -> list[float] | None:
        """Cached embedding of the thought text.

        Set by SemanticOscillationDetector after embedding. Returns None
        if no embedding has been computed yet.

        Returns:
            A list of floats representing the thought embedding, or None.
        """
        return self._thought_vector

    @thought_vector.setter
    def thought_vector(self, value: list[float] | None) -> None:
        """Set the cached thought embedding vector."""
        self._thought_vector = value

    @property
    def has_tool_call(self) -> bool:
        """Whether this step involves a tool call."""
        return self.action is not None

    @property
    def has_observation(self) -> bool:
        """Whether this step has a tool observation."""
        return self.observation is not None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the step to a dictionary.

        Note: The cached thought_vector is NOT included in serialization
        to keep output compact. Re-embed after deserialization if needed.
        """
        return {
            "step_number": self.step_number,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentStep:
        """Deserialize a step from a dictionary.

        Args:
            data: Dictionary with step fields.

        Returns:
            An AgentStep instance.
        """
        return cls(
            step_number=data["step_number"],
            thought=data["thought"],
            action=data.get("action"),
            action_input=data.get("action_input"),
            observation=data.get("observation"),
            tokens_used=data.get("tokens_used", 0),
            latency_ms=data.get("latency_ms", 0.0),
        )

    def __str__(self) -> str:
        """Human-readable representation of the step."""
        parts = [f"Step {self.step_number}"]
        if self.action:
            parts.append(f"Action: {self.action}")
            parts.append(f"Input: {self.action_input}")
        parts.append(f"Thought: {self.thought[:80]}{'...' if len(self.thought) > 80 else ''}")
        if self.observation:
            obs_preview = self.observation[:60]
            parts.append(f"Observation: {obs_preview}{'...' if len(self.observation) > 60 else ''}")
        parts.append(f"Tokens: {self.tokens_used}")
        return " | ".join(parts)
