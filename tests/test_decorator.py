"""Tests for the universal @guarded decorator."""

import asyncio
from unittest.mock import MagicMock

import pytest

from longguard import (
    AgentStep,
    CircuitBreakerTrippedError,
    guarded,
)


def test_guarded_sync_with_explicit_step():
    """Test decorating a function that receives an AgentStep."""

    @guarded(tool_repeat_threshold=2, max_reflections=1)
    def execute_step(step: AgentStep) -> str:
        return f"executed {step.step_number}"

    # Step 1: Clean step
    s1 = AgentStep(step_number=1, thought="Thinking 1", action="search", action_input={"q": "A"})
    res1 = execute_step(s1)
    assert res1 == "executed 1"
    assert len(execute_step.report.steps) == 1

    # Step 2: Repeat action -> triggers reflection
    s2 = AgentStep(step_number=2, thought="Thinking 2", action="search", action_input={"q": "A"})
    res2 = execute_step(s2)
    assert res2 == "executed 2"
    assert execute_step.last_reflection_prompt is not None

    # Step 3: Repeated again -> trips circuit breaker
    s3 = AgentStep(step_number=3, thought="Thinking 3", action="search", action_input={"q": "A"})
    with pytest.raises(CircuitBreakerTrippedError) as exc_info:
        execute_step(s3)

    assert "tool_repeat" in str(exc_info.value)
    assert exc_info.value.report is not None


def test_guarded_sync_with_kwargs():
    """Test decorating a function where kwargs define thought and tool."""

    @guarded(tool_repeat_threshold=2, max_reflections=1)
    def run_tool(
        thought: str,
        action: str,
        action_input: dict,
        observation: str = "",
        tokens_used: int = 10,
    ):
        return f"result of {action}"

    run_tool(thought="step 1", action="calc", action_input={"x": 1})
    run_tool(thought="step 2", action="calc", action_input={"x": 1})

    with pytest.raises(CircuitBreakerTrippedError) as exc_info:
        run_tool(thought="step 3", action="calc", action_input={"x": 1})

    assert "tool_repeat" in str(exc_info.value)


def test_guarded_on_kill_return():
    """Test on_kill='return' returning graceful fallback instead of raising."""

    @guarded(tool_repeat_threshold=2, max_reflections=1, on_kill="return")
    def tool_call(thought: str, action: str, action_input: dict):
        return "success"

    res1 = tool_call(thought="1", action="fetch", action_input={})
    assert res1 == "success"
    res2 = tool_call(thought="2", action="fetch", action_input={})
    assert res2 == "success"
    fallback = tool_call(thought="3", action="fetch", action_input={})

    assert isinstance(fallback, str)
    assert "Agent terminated by LongGuard" in fallback


def test_guarded_on_kill_custom_callable():
    """Test custom on_kill callback handler."""

    def custom_handler(decision, *args, **kwargs):
        return {"halted": True, "reason": decision.reason}

    @guarded(tool_repeat_threshold=2, max_reflections=1, on_kill=custom_handler)
    def tool_call(thought: str, action: str, action_input: dict):
        return {"halted": False}

    res1 = tool_call(thought="1", action="db", action_input={})
    assert res1["halted"] is False
    res2 = tool_call(thought="2", action="db", action_input={})
    assert res2["halted"] is False
    res3 = tool_call(thought="3", action="db", action_input={})

    assert res3["halted"] is True
    assert "tool_repeat" in res3["reason"]


def test_guarded_post_call_return_dict():
    """Test extracting step from function return value dictionary."""

    @guarded(tool_repeat_threshold=2, max_reflections=1)
    def generate_plan(prompt: str):
        return {
            "thought": f"planning for {prompt}",
            "action": "search",
            "action_input": {"query": "python"},
            "tokens_used": 50,
        }

    generate_plan("step 1")
    generate_plan("step 2")

    with pytest.raises(CircuitBreakerTrippedError):
        generate_plan("step 3")


def test_guarded_post_call_openai_response():
    """Test extracting step from mock OpenAI ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = "I will check the files."
    choice.message.tool_calls = None
    choice.finish_reason = "stop"

    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 50
    usage.total_tokens = 150

    mock_resp = MagicMock()
    mock_resp.choices = [choice]
    mock_resp.usage = usage

    @guarded(model="gpt-4o", max_cost_usd=0.0001)  # extremely low limit
    def call_gpt():
        return mock_resp

    # First call exceeds $0.0001 limit and trips immediately
    with pytest.raises(CircuitBreakerTrippedError) as exc_info:
        call_gpt()

    assert "cost_budget_exceeded" in str(exc_info.value)


@pytest.mark.asyncio
async def test_guarded_async_function():
    """Test decorating native Python async def functions."""

    @guarded(tool_repeat_threshold=2, max_reflections=1)
    async def async_step(thought: str, action: str, action_input: dict):
        await asyncio.sleep(0.001)
        return "async_ok"

    await async_step(thought="1", action="ping", action_input={})
    await async_step(thought="2", action="ping", action_input={})

    with pytest.raises(CircuitBreakerTrippedError):
        await async_step(thought="3", action="ping", action_input={})


def test_guarded_generator():
    """Test decorating a Python generator."""

    @guarded(tool_repeat_threshold=2, max_reflections=1)
    def step_generator():
        for i in range(10):
            yield AgentStep(
                step_number=i + 1,
                thought=f"iteration {i}",
                action="loop_tool",
                action_input={"i": 0},
            )

    gen = step_generator()
    next(gen)
    next(gen)

    with pytest.raises(CircuitBreakerTrippedError):
        next(gen)


def test_guarded_reset():
    """Test that wrapper.reset() clears breaker history."""

    @guarded(tool_repeat_threshold=2)
    def my_step(thought: str, action: str, action_input: dict):
        return "ok"

    my_step(thought="1", action="a", action_input={})
    assert len(my_step.report.steps) == 1

    my_step.reset()
    assert len(my_step.report.steps) == 0
    assert my_step.last_reflection_prompt is None
