"""Tests for LangChain integration."""

import sys
from unittest.mock import MagicMock

import pytest

from longguard.config import GuardConfig
from longguard.integrations.langchain import GuardedAgentExecutor, GuardTerminatedException


class MockBaseCallbackHandler:
    pass

sys.modules["langchain_core"] = MagicMock()
sys.modules["langchain_core.callbacks"] = MagicMock(BaseCallbackHandler=MockBaseCallbackHandler)

class MockAgentAction:
    def __init__(self, tool="test_tool", tool_input="test_input", log="thought"):
        self.tool = tool
        self.tool_input = tool_input
        self.log = log

class MockAgentFinish:
    def __init__(self, return_values={"output": "final answer"}, log="final thought"):
        self.return_values = return_values
        self.log = log

class MockExecutor:
    def invoke(self, inputs, **kwargs):
        # Simulate an execution that triggers callbacks
        callbacks = kwargs.get("callbacks", [])
        for cb in callbacks:
            if hasattr(cb, "on_agent_action"):
                cb.on_agent_action(MockAgentAction())
            if hasattr(cb, "on_tool_end"):
                cb.on_tool_end("tool result")
        return {"output": "success"}

def test_init():
    executor = MockExecutor()
    guard = GuardedAgentExecutor(executor)
    assert guard.config is not None
    assert guard.breaker is not None

def test_extract_step_from_action():
    guard = GuardedAgentExecutor(MockExecutor())
    action = MockAgentAction()
    step = guard._extract_step_from_action(action, "observation")
    assert step.step_number == 1
    assert step.thought == "thought"
    assert step.action == "test_tool"
    assert step.action_input == "test_input"
    assert step.observation == "observation"

def test_extract_step_from_finish():
    guard = GuardedAgentExecutor(MockExecutor())
    finish = MockAgentFinish()
    step = guard._extract_step_from_finish(finish)
    assert step.step_number == 1
    assert step.thought == "final thought"
    assert step.observation == "{'output': 'final answer'}"

def test_make_pivot_message():
    guard = GuardedAgentExecutor(MockExecutor())
    msg = guard._make_pivot_message("pivot prompt")
    assert msg == "\npivot prompt\n"

def test_run_success():
    guard = GuardedAgentExecutor(MockExecutor())
    result = guard.run("hello")
    assert result == "success"

def test_run_dict_return():
    class DictMockExecutor:
        def invoke(self, inputs, **kwargs):
            return {"output": "dict_success"}
    guard = GuardedAgentExecutor(DictMockExecutor())
    result = guard.run("hello")
    assert result == "dict_success"

def test_run_non_dict_return():
    class StringMockExecutor:
        def invoke(self, inputs, **kwargs):
            return "string_success"
    guard = GuardedAgentExecutor(StringMockExecutor())
    result = guard.run("hello")
    assert result == "string_success"

def test_invoke_kill():
    guard = GuardedAgentExecutor(MockExecutor(), GuardConfig(max_tokens_per_run=1))

    class BadExecutor:
        def invoke(self, inputs, **kwargs):
            callbacks = kwargs.get("callbacks", [])
            for cb in callbacks:
                if hasattr(cb, "on_agent_action"):
                    cb.on_agent_action(MockAgentAction(log="x" * 1000))
                if hasattr(cb, "on_tool_end"):
                    cb.on_tool_end("result")
            return {}

    guard._executor = BadExecutor()
    with pytest.raises(GuardTerminatedException):
        guard.invoke({"input": "test"})

def test_invoke_reflect():
    guard = GuardedAgentExecutor(MockExecutor(), GuardConfig(max_steps=10))
    decision = type(
        "Decision",
        (),
        {"action": "reflect", "reason": "test", "inject_prompt": "pivot", "report": None},
    )()
    guard.breaker.check = lambda x: decision

    # Should not raise exception
    guard.invoke({"input": "test"})

def test_invoke_recursion_error():
    class RecursionExecutor:
        def invoke(self, inputs, **kwargs):
            raise RecursionError("recursion limit hit")

    guard = GuardedAgentExecutor(RecursionExecutor())
    with pytest.raises(RecursionError):
        guard.invoke({"input": "test"})

def test_check_step_continue():
    guard = GuardedAgentExecutor(MockExecutor())
    decision = guard.check_step(MockAgentAction(), "obs")
    assert decision.action == "continue"

def test_check_step_kill():
    guard = GuardedAgentExecutor(MockExecutor(), GuardConfig(max_tokens_per_run=1))
    with pytest.raises(GuardTerminatedException):
        guard.check_step(MockAgentAction(log="x"*1000), "obs")

def test_get_report():
    guard = GuardedAgentExecutor(MockExecutor())
    assert guard.get_report() is not None

def test_reset():
    guard = GuardedAgentExecutor(MockExecutor())
    guard.check_step(MockAgentAction(), "obs")
    assert guard._step_counter == 1
    guard.reset()
    assert guard._step_counter == 0
