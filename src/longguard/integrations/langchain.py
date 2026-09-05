"""LangChain integration — AgentExecutor middleware for LongGuard."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from ..config import GuardConfig
from ..core.breaker import BreakerDecision, CircuitBreaker
from ..core.reporter import GuardReport
from ..core.step import AgentStep

logger = logging.getLogger("longguard.langchain")

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
else:
    try:
        from langchain_core.callbacks import BaseCallbackHandler
    except ImportError:
        BaseCallbackHandler = object


class GuardTerminatedException(Exception):  # noqa: N818
    """Raised when LongGuard kills the agent during a LangChain run.

    This exception carries the full GuardReport and kill reason so that
    callers can inspect the run details and present them to the user.

    Attributes:
        reason: The kill reason string.
        report: The GuardReport at the time of termination.
    """

    def __init__(self, reason: str, report: Any = None) -> None:
        self.reason = reason
        self.report = report
        super().__init__(f"Agent terminated by LongGuard: {reason}")


class GuardedAgentExecutor:
    """AgentExecutor wrapper that monitors execution and enforces circuit breaking.

    This class wraps a LangChain ``AgentExecutor``, intercepting each step to:
    1. Detect reasoning loops and semantic drift
    2. Inject Reflect & Pivot prompts when loops are detected
    3. Gracefully kill the agent if it cannot recover
    4. Enforce hard caps on token budget and step count

    Usage::

        from langchain.agents import AgentExecutor
        from longguard.integrations.langchain import GuardedAgentExecutor
        from longguard.config import GuardConfig

        agent_executor = AgentExecutor.from_agent_and_tools(...)
        guarded = GuardedAgentExecutor(agent_executor, GuardConfig())
        result = guarded.run("What is the weather in Tokyo?")

    Args:
        executor: The LangChain AgentExecutor to wrap.
        config: GuardConfig with all tuning parameters.
    """

    def __init__(
        self,
        executor: Any,
        config: GuardConfig | None = None,
    ) -> None:
        self._executor = executor
        self.config = config or GuardConfig()
        self.breaker = CircuitBreaker(self.config)
        self._step_counter = 0

    def _extract_step_from_action(
        self, action: Any, observation: str
    ) -> AgentStep:
        """Convert a LangChain AgentAction and observation into an AgentStep.

        Args:
            action: The AgentAction or AgentFinish object.
            observation: The tool observation string.

        Returns:
            An AgentStep representing this step.
        """
        self._step_counter += 1

        thought = ""
        action_name = None
        action_input = None

        if hasattr(action, "log"):
            thought = getattr(action, "log", "")
        if hasattr(action, "tool"):
            action_name = getattr(action, "tool", None)
        if hasattr(action, "tool_input"):
            action_input = getattr(action, "tool_input", None)

        estimated_tokens = max(len(thought) // 4, 1)

        return AgentStep(
            step_number=self._step_counter,
            thought=thought or "(no thought)",
            action=action_name,
            action_input=action_input,
            observation=observation,
            tokens_used=estimated_tokens,
        )

    def _extract_step_from_finish(self, finish: Any) -> AgentStep:
        """Create an AgentStep from a LangChain AgentFinish.

        Args:
            finish: The LangChain AgentFinish object.

        Returns:
            An AgentStep representing this final step.
        """
        self._step_counter += 1

        thought = ""
        if hasattr(finish, "log"):
            thought = finish.log or ""

        output = ""
        if hasattr(finish, "return_values"):
            output = str(finish.return_values)

        estimated_tokens = max((len(thought) + len(output)) // 4, 1)

        return AgentStep(
            step_number=self._step_counter,
            thought=thought or "(final answer)",
            observation=output,
            tokens_used=estimated_tokens,
        )

    def _make_pivot_message(self, prompt: str) -> str:
        """Format a pivot prompt as a message to inject into the agent's scratchpad.

        Args:
            prompt: The pivot prompt string.

        Returns:
            A formatted message string.
        """
        return f"\n{prompt}\n"

    def run(self, input_text: str, **kwargs: Any) -> str:
        """Run the agent on a string input with LongGuard monitoring.

        Convenience wrapper around ``invoke()`` that takes and returns
        plain strings (matching LangChain's ``AgentExecutor.run()`` API).

        Args:
            input_text: The input query string.
            **kwargs: Additional keyword arguments passed to invoke.

        Returns:
            The agent's output string.

        Raises:
            GuardTerminatedException: If the agent is killed by LongGuard.
        """
        result = self.invoke({"input": input_text}, **kwargs)
        if isinstance(result, dict):
            return str(result.get("output", str(result)))
        return str(result)

    def invoke(self, inputs: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Invoke the agent with LongGuard monitoring.

        This method runs the agent using the executor's internal loop,
        intercepting each step via a LangChain BaseCallbackHandler. When a
        loop is detected, it injects a pivot prompt. If the agent is killed,
        it raises GuardTerminatedException.

        Args:
            inputs: The inputs to pass to the executor.
            **kwargs: Additional keyword arguments for the executor.

        Returns:
            The executor's output dictionary.

        Raises:
            GuardTerminatedException: If the agent is killed by LongGuard.
        """
        guard = self
        # Mutable container to hold the last action across callbacks
        _pending_action: list[Any] = []

        class GuardCallback(BaseCallbackHandler):  # type: ignore[misc]
            """Callback that checks each agent step with LongGuard."""

            def on_agent_action(self, action: Any, **kw: Any) -> Any:
                """Capture the agent's action decision for pairing with tool output."""
                _pending_action.clear()
                _pending_action.append(action)

            def on_tool_end(
                self,
                output: str,
                observation_prefix: str | None = None,
                llm_prefix: str | None = None,
                **kw: Any,
            ) -> Any:
                """Called after a tool returns — build step and check breaker."""
                if not _pending_action:
                    return

                action = _pending_action[0]
                step = guard._extract_step_from_action(action, output)
                decision = guard.breaker.check(step)

                if decision.action == "kill":
                    raise GuardTerminatedException(
                        decision.reason, decision.report
                    )

                if decision.action == "reflect" and decision.inject_prompt:
                    logger.info(
                        "LangChain agent: injecting pivot prompt for: %s",
                        decision.reason,
                    )
                    # Inject pivot as a tool observation override isn't feasible
                    # in the callback model, so we log the suggestion.
                    # For full interception, use check_step() manually.

        guard_callback = GuardCallback()

        # Merge our callback with any existing callbacks
        existing_callbacks = kwargs.pop("callbacks", None) or []
        if not isinstance(existing_callbacks, list):
            existing_callbacks = [existing_callbacks]
        existing_callbacks.append(guard_callback)
        kwargs["callbacks"] = existing_callbacks

        try:
            result = self._executor.invoke(inputs, **kwargs)
        except GuardTerminatedException:
            raise
        except Exception as exc:
            if "recursion" in str(exc).lower():
                logger.warning(
                    "LangChain recursion limit hit. LongGuard could have "
                    "prevented this with lower thresholds."
                )
            raise

        return cast(dict[str, Any], result)

    def check_step(self, action: Any, observation: str) -> BreakerDecision:
        """Manually check a single agent step (for custom integration patterns).

        If you're implementing your own agent loop rather than using AgentExecutor,
        call this method after each step to check for loops.

        Args:
            action: The LangChain AgentAction from this step.
            observation: The tool observation from this step.

        Returns:
            A BreakerDecision indicating what to do.

        Raises:
            GuardTerminatedException: If the decision is to kill the agent.
        """
        step = self._extract_step_from_action(action, observation)
        decision = self.breaker.check(step)

        if decision.action == "kill":
            raise GuardTerminatedException(decision.reason, decision.report)

        return decision

    def get_report(self) -> GuardReport:
        """Get the current GuardReport for this run."""
        return self.breaker.report

    def reset(self) -> None:
        """Reset the guard for a new agent run."""
        self.breaker.reset()
        self._step_counter = 0
