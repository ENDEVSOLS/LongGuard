"""Tests for Strands integration."""

import pytest

from longguard.config import GuardConfig
from longguard.integrations.strands import GuardTerminatedError, StrandsGuard


class MockStrandsAgentWithOnStep:
    def __init__(self):
        self.on_step_called = False

    def on_step(self, event):
        self.on_step_called = True
        return "step completed"

class MockStrandsAgentWithAddCallback:
    def __init__(self):
        self.callbacks = []

    def add_callback(self, cb):
        self.callbacks.append(cb)

class MockStrandsAgentPlain:
    pass

def test_init():
    guard = StrandsGuard()
    assert guard.config is not None
    assert guard.breaker is not None

def test_extract_step():
    guard = StrandsGuard()
    event = {
        "type": "tool_call",
        "data": {
            "text": "thinking",
            "tool_name": "search",
            "tool_input": "query",
            "tool_result": "found something"
        }
    }
    step = guard._extract_step(event)
    assert step.step_number == 1
    assert step.thought == "thinking"
    assert step.action == "search"
    assert step.action_input == "query"
    assert step.observation == "found something"

def test_wrap_on_step():
    guard = StrandsGuard()
    agent = MockStrandsAgentWithOnStep()
    wrapped = guard.wrap(agent)

    assert wrapped is agent
    assert hasattr(agent, "__longguard__")

    event = {"type": "test", "data": {"text": "hello"}}
    result = agent.on_step(event)
    assert result == "step completed"
    assert agent.on_step_called is True

def test_wrap_on_step_kill():
    guard = StrandsGuard(GuardConfig(max_tokens_per_run=1))
    agent = MockStrandsAgentWithOnStep()
    guard.wrap(agent)

    event = {"type": "test", "data": {"text": "x" * 1000}}
    with pytest.raises(GuardTerminatedError):
        agent.on_step(event)

def test_wrap_on_step_reflect():
    guard = StrandsGuard(GuardConfig(max_steps=10))
    agent = MockStrandsAgentWithOnStep()
    guard.wrap(agent)

    decision = type(
        "Decision",
        (),
        {"action": "reflect", "reason": "test", "inject_prompt": "pivot", "report": None},
    )()
    guard.breaker.check = lambda x: decision

    event = {"type": "test", "data": {"text": "hello"}}
    agent.on_step(event)
    assert event["data"]["__longguard_pivot__"] == "pivot"

def test_wrap_add_callback():
    guard = StrandsGuard()
    agent = MockStrandsAgentWithAddCallback()
    guard.wrap(agent)

    assert len(agent.callbacks) == 1

    # Execute callback
    agent.callbacks[0]({"type": "test", "data": {"text": "hello"}})
    assert guard._step_counter == 1

def test_wrap_add_callback_kill():
    guard = StrandsGuard(GuardConfig(max_tokens_per_run=1))
    agent = MockStrandsAgentWithAddCallback()
    guard.wrap(agent)

    with pytest.raises(GuardTerminatedError):
        agent.callbacks[0]({"type": "test", "data": {"text": "x" * 1000}})

def test_wrap_fallback():
    guard = StrandsGuard()
    agent = MockStrandsAgentPlain()
    wrapped = guard.wrap(agent)
    assert hasattr(wrapped, "__longguard__")

def test_check_step_continue():
    guard = StrandsGuard()
    decision = guard.check_step({"type": "test", "data": {}})
    assert decision.action == "continue"

def test_check_step_kill():
    guard = StrandsGuard(GuardConfig(max_tokens_per_run=1))
    with pytest.raises(GuardTerminatedError):
        guard.check_step({"type": "test", "data": {"text": "x" * 1000}})

def test_get_report():
    guard = StrandsGuard()
    assert guard.get_report() is not None

def test_reset():
    guard = StrandsGuard()
    guard.check_step({"type": "test", "data": {}})
    assert guard._step_counter == 1
    guard.reset()
    assert guard._step_counter == 0
