"""CrewAI integration — Step and Crew circuit breaker middleware for LongGuard."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ..config import GuardConfig
from ..core.breaker import BreakerDecision, CircuitBreaker
from ..core.reporter import GuardReport
from ..core.step import AgentStep

logger = logging.getLogger("longguard.crewai")


class GuardTerminatedException(Exception):  # noqa: N818
    """Raised when LongGuard kills an agent during a CrewAI run.

    Attributes:
        reason: Human-readable kill reason string.
        report: The GuardReport at the time of termination.
        decision: The BreakerDecision that triggered the kill.
    """

    def __init__(
        self,
        reason: str,
        report: GuardReport | None = None,
        decision: BreakerDecision | None = None,
    ) -> None:
        self.reason = reason
        self.report = report
        self.decision = decision
        super().__init__(f"Agent terminated by LongGuard: {reason}")


def _extract_step_from_crewai(
    step_data: Any,
    step_number: int,
) -> AgentStep:
    """Extract an AgentStep from CrewAI's AgentAction, AgentFinish, or dict."""
    if isinstance(step_data, AgentStep):
        return step_data

    thought: str = ""
    action: str | None = None
    action_input: Any | None = None
    observation: str | None = None
    tokens_used: int = 0

    # 1. Dictionary representation
    if isinstance(step_data, dict):
        thought = str(step_data.get("thought", "") or step_data.get("log", ""))
        action = step_data.get("tool") or step_data.get("action")
        action_input = step_data.get("tool_input") or step_data.get("action_input")
        observation = step_data.get("result") or step_data.get("observation")
        tokens_used = int(step_data.get("tokens_used", 0) or step_data.get("tokens", 0))

    # 2. CrewAI AgentAction / AgentFinish object
    elif hasattr(step_data, "tool") or hasattr(step_data, "output"):
        # AgentAction has tool and tool_input
        if hasattr(step_data, "tool"):
            action = getattr(step_data, "tool", None)
            action_input = getattr(step_data, "tool_input", None)
        # Thought or reasoning log
        if hasattr(step_data, "thought"):
            thought = str(getattr(step_data, "thought") or "")
        elif hasattr(step_data, "log"):
            thought = str(getattr(step_data, "log") or "")
        elif hasattr(step_data, "text"):
            thought = str(getattr(step_data, "text") or "")

        # Result / observation
        if hasattr(step_data, "result"):
            observation = str(getattr(step_data, "result") or "")
        elif hasattr(step_data, "observation"):
            observation = str(getattr(step_data, "observation") or "")
        elif hasattr(step_data, "output"):
            observation = str(getattr(step_data, "output") or "")

        if hasattr(step_data, "tokens_used"):
            tokens_used = int(getattr(step_data, "tokens_used") or 0)

    # 3. Tuple of (AgentAction, observation)
    elif isinstance(step_data, (tuple, list)) and len(step_data) >= 1:
        first = step_data[0]
        if hasattr(first, "tool"):
            action = getattr(first, "tool", None)
            action_input = getattr(first, "tool_input", None)
            thought = str(getattr(first, "log", "") or getattr(first, "thought", ""))
        if len(step_data) >= 2:
            observation = str(step_data[1])

    # 4. Fallback string representation
    else:
        thought = str(step_data)

    return AgentStep(
        step_number=step_number,
        thought=thought,
        action=action,
        action_input=action_input,
        observation=observation,
        tokens_used=tokens_used,
    )


class CrewGuard:
    """Circuit breaker manager for CrewAI agents and crews.

    Provides a `step_callback` method that hooks directly into CrewAI's
    agent and crew execution loop, monitoring in-flight tool calls and thoughts.

    Usage::

        from crewai import Agent, Crew, Task
        from longguard import GuardConfig
        from longguard.integrations.crewai import CrewGuard

        guard = CrewGuard(GuardConfig(max_cost_usd=0.50, tool_repeat_threshold=3))

        # Attach to an individual agent:
        researcher = Agent(
            role="Researcher",
            goal="Find market data",
            backstory="...",
            step_callback=guard.step_callback,
        )

        # Or attach to an entire Crew:
        crew = Crew(
            agents=[researcher],
            tasks=[task],
            step_callback=guard.step_callback,
        )

        # Access telemetry afterwards:
        print(guard.summary())

    Args:
        config: GuardConfig with detection thresholds and caps.
        breaker: Optional pre-configured CircuitBreaker instance.
    """

    def __init__(
        self,
        config: GuardConfig | None = None,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self.config = config or GuardConfig()
        self.breaker = breaker or CircuitBreaker(self.config)
        self.last_reflection: str | None = None
        self._previous_step_callback: Callable[[Any], Any] | None = None

    @property
    def report(self) -> GuardReport:
        """The GuardReport of the underlying CircuitBreaker."""
        return self.breaker.report

    def get_report(self) -> GuardReport:
        """Return the GuardReport."""
        return self.breaker.report

    def summary(self) -> str:
        """Return human-readable summary of the run telemetry."""
        return self.breaker.report.summary()

    def reset(self) -> None:
        """Reset breaker state and step history for a new crew run."""
        self.breaker.reset()
        self.last_reflection = None

    def step_callback(self, step_output: Any) -> None:
        """CrewAI step callback hook.

        Inspects the step, evaluates circuit breaker conditions, and terminates
        with `GuardTerminatedException` if a loop or hard cap is tripped.
        """
        step_number = len(self.breaker.report.steps) + 1
        agent_step = _extract_step_from_crewai(step_output, step_number=step_number)

        decision = self.breaker.check(agent_step)

        if decision.action == "reflect":
            self.last_reflection = decision.inject_prompt
            logger.warning(
                "[LongGuard] Reflection triggered on step %d: %s",
                step_number,
                decision.reason,
            )
        elif decision.action == "kill":
            logger.error(
                "[LongGuard] Circuit tripped on step %d: %s",
                step_number,
                decision.reason,
            )
            raise GuardTerminatedException(
                reason=decision.reason,
                report=self.breaker.report,
                decision=decision,
            )

        # Chain previous callback if attached
        if self._previous_step_callback is not None:
            self._previous_step_callback(step_output)


def add_guard_to_crew(
    crew: Any,
    config: GuardConfig | None = None,
    guard: CrewGuard | None = None,
) -> Any:
    """Attach LongGuard protection to all agents in a CrewAI Crew in 1 line.

    Usage::

        from crewai import Crew
        from longguard.integrations.crewai import add_guard_to_crew

        crew = Crew(agents=[...], tasks=[...])
        crew = add_guard_to_crew(crew)
        result = crew.kickoff()

        # Telemetry:
        print(crew.__longguard__.summary())

    Args:
        crew: CrewAI Crew instance.
        config: GuardConfig settings.
        guard: Optional existing CrewGuard instance.

    Returns:
        The same crew instance with LongGuard attached.
    """
    g = guard or CrewGuard(config=config)

    # 1. Attach to crew step_callback
    existing_crew_cb = getattr(crew, "step_callback", None)
    if existing_crew_cb is not None and existing_crew_cb != g.step_callback:
        g._previous_step_callback = existing_crew_cb
    setattr(crew, "step_callback", g.step_callback)

    # 2. Attach to all agents in the crew
    agents = getattr(crew, "agents", None)
    if agents and isinstance(agents, (list, tuple)):
        for agent in agents:
            existing_agent_cb = getattr(agent, "step_callback", None)
            if existing_agent_cb is None or existing_agent_cb != g.step_callback:
                setattr(agent, "step_callback", g.step_callback)

    # 3. Store reference for easy retrieval
    setattr(crew, "__longguard__", g)
    return crew
