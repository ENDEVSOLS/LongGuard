"""ReflectAndPivotInjector — generates intervention prompts when loops are detected."""

from __future__ import annotations

import uuid
from typing import Any

from .detectors.base import DetectionResult
from .step import AgentStep

# Unique sentinel prefix to distinguish genuine LongGuard interventions
# from user-crafted prompt injection attempts.
_SENTINEL = f"[LONGGUARD-{uuid.uuid4().hex[:8].upper()}]"

# Default pivot prompt templates.
# Available template variables:
#   {tool}              — tool name (tool_repeat)
#   {count}             — number of times repeated (tool_repeat)
#   {hash}              — action hash fingerprint (tool_repeat)
#   {window}            — analysis window size (semantic_oscillation)
#   {thought_variance}  — measured thought variance (semantic_oscillation)
#   {steps}             — steps without progress (dead_end_drift)
#   {velocity}          — current token velocity (token_velocity)
#   {baseline}          — baseline token velocity (token_velocity)
PIVOT_TEMPLATES: dict[str, str] = {
    "tool_repeat": (
        "SYSTEM OVERRIDE — LOOP DETECTED:\n"
        "You have called the tool `{tool}` with identical arguments {count} times consecutively.\n"
        "This approach is not producing new information.\n\n"
        "REQUIRED: Stop. Reflect. Then choose ONE of:\n"
        "1. Try a COMPLETELY DIFFERENT tool to get the information you need\n"
        "2. Re-scope your goal — the information may not be retrievable via tools\n"
        "3. Provide a partial answer with what you already know and explain what's missing\n\n"
        "Do NOT repeat the same tool call again."
    ),
    "semantic_oscillation": (
        "SYSTEM OVERRIDE — REASONING LOOP DETECTED:\n"
        "Your recent reasoning steps are semantically identical — you are cycling "
        "through the same thoughts without making progress.\n\n"
        "REQUIRED: Take a fundamentally different approach:\n"
        "1. Summarize what you have established with CERTAINTY so far\n"
        "2. Identify the SINGLE most critical unknown blocking your answer\n"
        "3. Address ONLY that unknown with your next action\n\n"
        "Do NOT repeat any reasoning pattern from your last {window} steps."
    ),
    "dead_end_drift": (
        "SYSTEM OVERRIDE — DEAD END DETECTED:\n"
        "Your last {steps} actions have not contributed new information toward the goal.\n\n"
        "REQUIRED: Either provide your BEST ANSWER with current information and explicitly "
        "state what is missing, or ABANDON this line of reasoning and try a categorically "
        "different approach. Do not continue the current path."
    ),
    "token_velocity": (
        "SYSTEM OVERRIDE — COST SPIKE DETECTED:\n"
        "Your token consumption velocity has spiked to {velocity:.0f} tokens/step, "
        "which is {ratio:.1f}x the normal baseline of {baseline:.0f} tokens/step.\n"
        "This typically indicates a runaway loop with growing context.\n\n"
        "REQUIRED: Immediately reduce context usage:\n"
        "1. Do NOT include previous full conversation history in your reasoning\n"
        "2. Summarize what you know in 2-3 sentences and proceed from there\n"
        "3. If you cannot answer concisely, provide your best partial answer now"
    ),
}


class ReflectAndPivotInjector:
    """Generates Reflect & Pivot intervention prompts from detection results.

    When a loop pattern is detected, the CircuitBreaker calls this injector to
    produce a prompt that forces the agent to stop, reflect, and change approach.
    The injector fills a template with evidence from the DetectionResult.

    Custom templates can be provided via the constructor or the GuardConfig.
    Template variables use Python's ``str.format_map()`` syntax with curly braces.

    Args:
        templates: Optional dict of custom templates to override defaults.
            Keys should be pattern names (e.g. ``"tool_repeat"``). Values are
            template strings with ``{variable}`` placeholders.
    """

    def __init__(self, templates: dict[str, str] | None = None) -> None:
        self._templates: dict[str, str] = {**PIVOT_TEMPLATES}
        if templates:
            self._templates.update(templates)

    def generate(self, result: DetectionResult, step: AgentStep) -> str:
        """Generate a Reflect & Pivot prompt for the given detection result.

        Selects the appropriate template based on ``result.pattern`` and fills
        it with evidence from the detection result. If no template exists for
        the pattern, a generic fallback prompt is used.

        Args:
            result: The DetectionResult that triggered the intervention.
            step: The AgentStep where the loop was detected.

        Returns:
            A formatted prompt string to inject into the agent's context.
        """
        template = self._templates.get(result.pattern)

        if template is None:
            return self._generate_fallback(result, step)

        # Build the template variable map from evidence.
        # Only known, safe keys are used — custom detector evidence
        # is NOT passed through to prevent format-string injection.
        variables: dict[str, Any] = {
            "tool": str(step.action or "unknown")[:200],
            "count": result.evidence.get("repeated_times", 0),
            "hash": str(result.evidence.get("hash", ""))[:16],
            "window": result.evidence.get("window_size", 0),
            "thought_variance": result.evidence.get("thought_variance", 0.0),
            "steps": result.evidence.get("steps_without_progress", 0),
            "velocity": result.evidence.get("current_velocity", 0.0),
            "baseline": result.evidence.get("baseline_velocity", 0.0),
            "ratio": result.evidence.get("velocity_ratio", 0.0),
        }

        # Format safely — missing keys are preserved as-is instead of KeyError
        class SafeDict(dict[str, Any]):
            def __missing__(self, key: str) -> str:
                return f"{{{key}}}"

        try:
            body = template.format_map(SafeDict(variables))
        except (ValueError, KeyError):
            body = template

        return f"{_SENTINEL} {body}"

    def _generate_fallback(self, result: DetectionResult, step: AgentStep) -> str:
        """Generate a generic fallback prompt when no template matches.

        Raw evidence is NOT injected into the prompt to prevent leaking
        internal state (file paths, PII, etc.) into the LLM context.

        Args:
            result: The DetectionResult.
            step: The AgentStep.

        Returns:
            A generic intervention prompt.
        """
        return (
            f"{_SENTINEL} SYSTEM OVERRIDE — LOOP PATTERN DETECTED ({result.pattern}):\n"
            f"Confidence: {result.confidence:.0%}\n\n"
            "REQUIRED: Stop your current approach. Reflect on what you have tried, "
            "and take a fundamentally different path. If you cannot make progress, "
            "provide the best answer you can with current information."
        )

    def add_template(self, pattern: str, template: str) -> None:
        """Register a custom template for a pattern.

        Args:
            pattern: The pattern name to associate with this template.
            template: A template string with ``{variable}`` placeholders.
        """
        self._templates[pattern] = template

    def get_template(self, pattern: str) -> str | None:
        """Get the template for a pattern, if one exists.

        Args:
            pattern: The pattern name.

        Returns:
            The template string, or None if no template is registered.
        """
        return self._templates.get(pattern)

    @property
    def templates(self) -> dict[str, str]:
        """Return a copy of all registered templates."""
        return dict(self._templates)
