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
    def from_openai_response(
        cls,
        response: Any,
        step_number: int,
        *,
        observation: str | None = None,
        latency_ms: float = 0.0,
    ) -> AgentStep:
        """Create an AgentStep from a raw OpenAI chat completion response object.

        Works with any object that follows the OpenAI Python SDK response shape,
        including ``openai.types.chat.ChatCompletion``.  No ``openai`` package
        import is required — duck-typing is used throughout.

        Args:
            response: A raw OpenAI ``ChatCompletion`` response object.
            step_number: Sequential step index (1-based).
            observation: Tool observation from the *previous* step, if any.
            latency_ms: Wall-clock time for this step in milliseconds.

        Returns:
            An :class:`AgentStep` populated from the response.

        Example::

            import openai
            from longguard import AgentStep, CircuitBreaker, GuardConfig

            client = openai.OpenAI()
            breaker = CircuitBreaker(GuardConfig(model="gpt-4o"))

            for i in range(1, 31):
                response = client.chat.completions.create(...)
                step = AgentStep.from_openai_response(response, step_number=i)
                decision = breaker.check(step)
                if decision.action == "kill":
                    break
        """
        choice = response.choices[0]
        message = choice.message

        # --- Thought: extract text content ---
        content = message.content or ""
        if not isinstance(content, str):
            # Multimodal: concatenate text parts
            text_parts = [
                part.text
                for part in content
                if hasattr(part, "type") and part.type == "text" and hasattr(part, "text")
            ]
            content = " ".join(text_parts)

        # --- Action: first tool call (if any) ---
        action: str | None = None
        action_input: Any | None = None
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            first = tool_calls[0]
            fn = getattr(first, "function", None)
            if fn is not None:
                action = getattr(fn, "name", None)
                raw_args = getattr(fn, "arguments", None)
                if raw_args:
                    try:
                        import json as _json
                        action_input = _json.loads(raw_args)
                    except Exception:
                        action_input = raw_args

        # --- Token usage ---
        tokens_used = 0
        usage = getattr(response, "usage", None)
        if usage is not None:
            tokens_used = getattr(usage, "total_tokens", 0) or 0

        return cls(
            step_number=step_number,
            thought=content,
            action=action,
            action_input=action_input,
            observation=observation,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
        )

    @classmethod
    def from_anthropic_response(
        cls,
        response: Any,
        step_number: int,
        *,
        observation: str | None = None,
        latency_ms: float = 0.0,
    ) -> AgentStep:
        """Create an AgentStep from a raw Anthropic Messages API response object.

        Works with any object that follows the Anthropic Python SDK response shape,
        including ``anthropic.types.Message``.  No ``anthropic`` package import
        is required — duck-typing is used throughout.

        Args:
            response: A raw Anthropic ``Message`` response object.
            step_number: Sequential step index (1-based).
            observation: Tool observation from the *previous* step, if any.
            latency_ms: Wall-clock time for this step in milliseconds.

        Returns:
            An :class:`AgentStep` populated from the response.

        Example::

            import anthropic
            from longguard import AgentStep, CircuitBreaker, GuardConfig

            client = anthropic.Anthropic()
            breaker = CircuitBreaker(GuardConfig(model="claude-3-5-sonnet"))

            for i in range(1, 31):
                response = client.messages.create(...)
                step = AgentStep.from_anthropic_response(response, step_number=i)
                decision = breaker.check(step)
                if decision.action == "kill":
                    break
        """
        content_blocks = getattr(response, "content", []) or []

        # --- Thought: collect all text blocks ---
        thought_parts: list[str] = []
        action: str | None = None
        action_input: Any | None = None

        for block in content_blocks:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text = getattr(block, "text", "") or ""
                if text:
                    thought_parts.append(text)
            elif block_type == "tool_use" and action is None:
                # First tool_use block wins
                action = getattr(block, "name", None)
                action_input = getattr(block, "input", None)

        thought = " ".join(thought_parts)

        # --- Token usage ---
        tokens_used = 0
        usage = getattr(response, "usage", None)
        if usage is not None:
            input_toks = getattr(usage, "input_tokens", 0) or 0
            output_toks = getattr(usage, "output_tokens", 0) or 0
            tokens_used = input_toks + output_toks

        return cls(
            step_number=step_number,
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
        )

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
