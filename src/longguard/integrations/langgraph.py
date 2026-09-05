"""LangGraph integration — wrap StateGraph nodes with LongGuard hooks."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

from ..config import GuardConfig
from ..core.breaker import CircuitBreaker
from ..core.reporter import GuardReport
from ..core.step import AgentStep

logger = logging.getLogger("longguard.langgraph")


def _has_langchain_core() -> bool:
    """Check if langchain-core is available."""
    try:
        import langchain_core.messages  # noqa: F401
        return True
    except ImportError:
        return False


class LongGuard:
    """Main orchestrator that wraps LangGraph nodes with guard hooks.

    This class manages a CircuitBreaker and provides methods to wrap
    individual LangGraph node functions so that every step is monitored.
    On loop detection, it injects Reflect & Pivot prompts into the agent's
    state. On kill, it gracefully terminates the agent by adding a final
    message and termination flags.

    Usage::

        from longguard.integrations.langgraph import LongGuard
        from longguard.config import GuardConfig

        guard = LongGuard(GuardConfig())
        guarded_fn = guard.wrap_node("agent", original_agent_fn)

    Args:
        config: GuardConfig with all tuning parameters. Defaults are used
            if not provided.
    """

    def __init__(self, config: GuardConfig | None = None) -> None:
        self.config = config or GuardConfig()
        self.breaker = CircuitBreaker(self.config)
        self._step_counter = 0

    def _extract_step(self, state: dict[str, Any]) -> AgentStep:
        """Convert a LangGraph state dict to an AgentStep.

        LangGraph states typically have a ``"messages"`` key with a list of
        message objects. The last AI message is used for the thought/action,
        and the last tool message is used for the observation.

        Args:
            state: The LangGraph state dictionary.

        Returns:
            An AgentStep representing the current state.
        """
        self._step_counter += 1
        messages = state.get("messages", [])

        thought = ""
        action = None
        action_input = None
        observation = None

        # Extract from the message history
        if messages:
            # Look for the last AI message to extract thought and action
            ai_msg = None
            for msg in reversed(messages):
                if isinstance(msg, dict):
                    has_add_tool_calls = (
                        isinstance(msg.get("additional_kwargs"), dict)
                        and bool(msg["additional_kwargs"].get("tool_calls"))
                    )
                    if msg.get("type") == "ai" or bool(msg.get("tool_calls")) or has_add_tool_calls:
                        ai_msg = msg
                        break
                else:
                    has_add_tool_calls = (
                        hasattr(msg, "additional_kwargs")
                        and isinstance(msg.additional_kwargs, dict)
                        and bool(msg.additional_kwargs.get("tool_calls"))
                    )
                    is_ai = getattr(msg, "type", "") == "ai"
                    has_tc = bool(getattr(msg, "tool_calls", None))
                    if is_ai or has_tc or has_add_tool_calls:
                        ai_msg = msg
                        break

            # Fallback: if no explicit AI message found, pick the last non-tool message
            if ai_msg is None:
                for msg in reversed(messages):
                    if isinstance(msg, dict):
                        m_type = msg.get("type", "")
                    else:
                        m_type = getattr(msg, "type", "")
                    if m_type != "tool":
                        ai_msg = msg
                        break

            if ai_msg is not None:
                try:
                    if isinstance(ai_msg, dict):
                        content = ai_msg.get("content", "")
                        tcs = ai_msg.get("tool_calls", [])
                    else:
                        content = getattr(ai_msg, "content", "")
                        tcs = getattr(ai_msg, "tool_calls", [])

                    if content:
                        if isinstance(content, list):
                            texts = []
                            for item in content:
                                if isinstance(item, dict) and "text" in item:
                                    texts.append(item["text"])
                                elif isinstance(item, str):
                                    texts.append(item)
                            thought = "\n".join(texts)
                        else:
                            thought = str(content)

                    if tcs:
                        tc = tcs[0]
                        action = tc.get("name")
                        action_input = tc.get("args")
                    else:
                        add_kwargs = (
                            ai_msg.get("additional_kwargs", {})
                            if isinstance(ai_msg, dict)
                            else getattr(ai_msg, "additional_kwargs", {})
                        )
                        if isinstance(add_kwargs, dict):
                            tool_calls = add_kwargs.get("tool_calls", [])
                            if tool_calls:
                                tc = tool_calls[0]
                                func = tc.get("function", {})
                                action = func.get("name")
                                import json
                                try:
                                    action_input = json.loads(func.get("arguments", "{}"))
                                except (json.JSONDecodeError, TypeError):
                                    action_input = func.get("arguments")
                except Exception:
                    pass

            # Look for observation in the last tool message
            for msg in reversed(messages):
                try:
                    if isinstance(msg, dict):
                        msg_type = msg.get("type", "")
                        obs_content = msg.get("content", None)
                    else:
                        msg_type = getattr(msg, "type", "")
                        obs_content = getattr(msg, "content", None)

                    if msg_type == "tool":
                        if isinstance(obs_content, list):
                            texts = []
                            for item in obs_content:
                                if isinstance(item, dict) and "text" in item:
                                    texts.append(item["text"])
                                elif isinstance(item, str):
                                    texts.append(item)
                            observation = "\n".join(texts)
                        else:
                            observation = str(obs_content) if obs_content else ""
                        break
                except Exception:
                    break

        # Estimate tokens from thought length (rough: 1 token ≈ 4 chars)
        estimated_tokens = max(len(thought) // 4, 1)

        return AgentStep(
            step_number=self._step_counter,
            thought=thought or "(no thought)",
            action=action,
            action_input=action_input,
            observation=observation,
            tokens_used=estimated_tokens,
        )

    def _inject_pivot(self, state: dict[str, Any], prompt: str) -> dict[str, Any]:
        """Inject a Reflect & Pivot prompt into the agent state.

        Adds the pivot prompt as a SystemMessage at the end of the messages
        list, ensuring the agent sees it before its next step. If langchain-core
        is not installed, the prompt is added as a plain dict message.

        Args:
            state: The current LangGraph state dictionary.
            prompt: The pivot prompt string to inject.

        Returns:
            The modified state dictionary with the pivot prompt injected.
        """
        messages = list(state.get("messages", []))

        if _has_langchain_core():
            from langchain_core.messages import SystemMessage
            messages.append(SystemMessage(content=prompt))
        else:
            # Fallback: add as a plain dict so tests without langchain-core work
            messages.append({"type": "system", "content": prompt})

        return {**state, "messages": messages}

    def wrap_node(self, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a LangGraph node function with guard hooks."""

        def guarded_fn(state: dict[str, Any], config: Any = None) -> dict[str, Any]:
            # Check if already terminated by a previous guard check
            if state.get("__longguard_terminated__"):
                return state

            # Extract step from current state
            step = self._extract_step(state)

            # Check with the circuit breaker
            decision = self.breaker.check(step)

            if decision.action == "kill":
                # Graceful termination — add final message to state
                logger.warning(
                    "Node '%s': Agent killed. Reason: %s", name, decision.reason
                )
                messages = list(state.get("messages", []))

                if _has_langchain_core():
                    from langchain_core.messages import AIMessage
                    messages.append(
                        AIMessage(
                            content=(
                                f"[LongGuard] Agent terminated: {decision.reason}. "
                                "The agent was stuck in a reasoning loop and could not "
                                "recover after reflection attempts."
                            )
                        )
                    )
                else:
                    # Fallback without langchain-core
                    messages.append({
                        "type": "ai",
                        "content": (
                            f"[LongGuard] Agent terminated: {decision.reason}. "
                            "The agent was stuck in a reasoning loop and could not "
                            "recover after reflection attempts."
                        ),
                    })

                return {
                    **state,
                    "messages": messages,
                    "__longguard_terminated__": True,
                    "__longguard_reason__": decision.reason,
                }

            if decision.action == "reflect":
                # Inject pivot prompt into state before running the node
                logger.info(
                    "Node '%s': Injecting Reflect & Pivot prompt for pattern: %s",
                    name,
                    decision.reason,
                )
                state = self._inject_pivot(state, decision.inject_prompt or "")

            # Run the original node function
            res: Any
            if hasattr(fn, "invoke"):
                if config is not None:
                    res = fn.invoke(state, config)
                else:
                    res = fn.invoke(state)
            elif config is not None:
                try:
                    res = fn(state, config)
                except TypeError:
                    res = fn(state)
            else:
                res = fn(state)

            return cast(dict[str, Any], res)

        # Preserve the original function's metadata
        guarded_fn.__name__ = f"guarded_{name}"
        setattr(guarded_fn, "__longguard_original__", fn)
        setattr(guarded_fn, "__longguard_name__", name)

        return guarded_fn

    def get_report(self) -> GuardReport:
        """Get the current GuardReport for this run."""
        return self.breaker.report

    def reset(self) -> None:
        """Reset the guard for a new agent run."""
        self.breaker.reset()
        self._step_counter = 0


def add_guard_to_graph(
    graph: Any,
    config: GuardConfig | None = None,
    exclude_nodes: list[str] | None = None,
) -> Any:
    """Wrap every node in a LangGraph StateGraph with LongGuard hooks."""
    guard = LongGuard(config)

    if exclude_nodes is None:
        # By default, exclude common non-agent nodes so LongGuard only evaluates
        # on true reasoning steps, avoiding double-counting and state machine resets.
        exclude_nodes = ["tools", "action", "__start__", "__end__"]

    # Access the internal nodes dict of the StateGraph
    nodes_dict = None
    if hasattr(graph, "nodes"):
        nodes_dict = graph.nodes
    elif hasattr(graph, "_nodes"):
        nodes_dict = graph._nodes

    if nodes_dict is None:
        logger.warning(
            "Could not find nodes dict in graph. "
            "Ensure you are passing a valid StateGraph instance."
        )
        return graph

    # Wrap each node
    for node_name in list(nodes_dict.keys()):
        if node_name in exclude_nodes:
            continue

        node_obj = nodes_dict[node_name]

        # Check if it's a modern LangGraph NodeSpec object
        if hasattr(node_obj, "runnable"):
            original_fn = node_obj.runnable
            wrapped = guard.wrap_node(node_name, original_fn)
            try:
                from langchain_core.runnables import RunnableLambda
                node_obj.runnable = RunnableLambda(wrapped)
            except ImportError:
                node_obj.runnable = wrapped
        else:
            # Fallback for older LangGraph versions
            original_fn = node_obj
            nodes_dict[node_name] = guard.wrap_node(node_name, original_fn)

    # Store guard reference on the graph for later access
    setattr(graph, "__longguard__", guard)

    return graph
