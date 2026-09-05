"""Strands Agents adapter — DRAFT / UNVERIFIED integration.

.. warning::
    This adapter is a **draft implementation** targeting a speculative API
    surface.  The ``strands-agents`` package is **not currently available** as a
    public Python package.  The ``on_step`` / ``add_callback`` hooks assumed
    here have not been validated against a real SDK.

    Use ``check_step()`` for manual integration with any callback-based agent
    framework, or wait for an official Strands SDK release before relying on
    ``wrap()``.

This adapter follows the same pattern as the LangGraph and LangChain
integrations: observe each step, detect loops, inject pivots, and kill
when necessary.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ..config import GuardConfig
from ..core.breaker import CircuitBreaker
from ..core.reporter import GuardReport
from ..core.step import AgentStep

logger = logging.getLogger("longguard.strands")


class StrandsGuard:
    """Guard wrapper for Strands Agents.

    This adapter provides hooks for the Strands Agents SDK, allowing
    LongGuard to monitor agent execution steps. Since Strands uses
    a callback-based architecture, this guard registers callbacks
    for each step.

    Usage::

        from longguard.integrations.strands import StrandsGuard
        from longguard.config import GuardConfig

        guard = StrandsGuard(GuardConfig())
        # Register guard callbacks with your Strands agent
        guarded_agent = guard.wrap(agent)

    Args:
        config: Optional GuardConfig. Uses defaults if not provided.
    """

    def __init__(self, config: GuardConfig | None = None) -> None:
        self.config = config or GuardConfig()
        self.breaker = CircuitBreaker(self.config)
        self._step_counter = 0

    def _extract_step(self, event: dict[str, Any]) -> AgentStep:
        """Convert a Strands event to an AgentStep.

        Strands events typically have a structure like:
        {
            "type": "tool_call" | "tool_result" | "text" | ...,
            "data": { ... }
        }

        Args:
            event: The Strands event dictionary.

        Returns:
            An AgentStep representing the event.
        """
        self._step_counter += 1

        event_type = event.get("type", "unknown")
        data = event.get("data", {})

        thought = str(data.get("text", data.get("thought", "")))
        action = data.get("tool_name", data.get("action"))
        action_input = data.get("tool_input", data.get("action_input"))
        observation = data.get("tool_result", data.get("observation"))

        # Estimate tokens
        text_len = len(thought) + len(str(observation or ""))
        estimated_tokens = max(text_len // 4, 1)

        return AgentStep(
            step_number=self._step_counter,
            thought=thought or f"(Strands event: {event_type})",
            action=action,
            action_input=action_input,
            observation=observation,
            tokens_used=estimated_tokens,
        )

    def wrap(self, agent: Any) -> Any:
        """Wrap a Strands agent with guard callbacks.

        This method attempts to register guard callbacks with the agent.
        Since the Strands API is still evolving, the exact mechanism
        may vary. This implementation uses the callback registration
        pattern if available.

        Args:
            agent: A Strands agent instance.

        Returns:
            The wrapped agent (may be the same instance with callbacks registered).
        """
        guard = self

        # Try to register a step callback
        if hasattr(agent, "on_step"):
            original_on_step = agent.on_step

            def guarded_on_step(event: dict[str, Any]) -> Any:
                step = guard._extract_step(event)
                decision = guard.breaker.check(step)

                if decision.action == "kill":
                    logger.warning(
                        "Strands agent killed by LongGuard: %s", decision.reason
                    )
                    raise GuardTerminatedError(decision.reason, decision.report)

                if decision.action == "reflect":
                    # Inject pivot prompt into the event data
                    logger.info(
                        "Strands agent: injecting pivot prompt for pattern: %s",
                        decision.reason,
                    )
                    if isinstance(event.get("data"), dict):
                        event["data"]["__longguard_pivot__"] = decision.inject_prompt

                # Call original callback
                if callable(original_on_step):
                    return original_on_step(event)

            agent.on_step = guarded_on_step

        elif hasattr(agent, "add_callback"):
            # Alternative: register as a callback
            agent.add_callback(self._make_callback())

        else:
            logger.warning(
                "Could not register guard callbacks with Strands agent. "
                "The agent does not support on_step or add_callback. "
                "Use check_step() manually in your agent loop."
            )

        setattr(agent, "__longguard__", guard)
        return agent

    def _make_callback(self) -> Callable[[dict[str, Any]], None]:
        """Create a callback function for Strands agents that support add_callback.

        Returns:
            A callback function that checks each step with LongGuard.
        """
        guard = self

        def callback(event: dict[str, Any]) -> None:
            step = guard._extract_step(event)
            decision = guard.breaker.check(step)

            if decision.action == "kill":
                raise GuardTerminatedError(decision.reason, decision.report)

            if decision.action == "reflect":
                logger.info(
                    "Strands agent: injecting pivot prompt for pattern: %s",
                    decision.reason,
                )

        return callback

    def check_step(self, event: dict[str, Any]) -> Any:
        """Manually check a Strands event (for custom integration patterns).

        If you're implementing your own agent loop, call this method after
        each step to check for loops.

        Args:
            event: The Strands event dictionary.

        Returns:
            A BreakerDecision indicating what to do.

        Raises:
            GuardTerminatedError: If the decision is to kill the agent.
        """
        step = self._extract_step(event)
        decision = self.breaker.check(step)

        if decision.action == "kill":
            raise GuardTerminatedError(decision.reason, decision.report)

        return decision

    def get_report(self) -> GuardReport:
        """Get the current GuardReport for this run."""
        return self.breaker.report

    def reset(self) -> None:
        """Reset the guard for a new agent run."""
        self.breaker.reset()
        self._step_counter = 0


class GuardTerminatedError(Exception):
    """Raised when LongGuard kills a Strands agent.

    Attributes:
        reason: The kill reason string.
        report: The GuardReport at the time of termination.
    """

    def __init__(self, reason: str, report: Any = None) -> None:
        self.reason = reason
        self.report = report
        super().__init__(f"Agent terminated by LongGuard: {reason}")
