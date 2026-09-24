"""Tests for CrewAI integration (CrewGuard and add_guard_to_crew)."""

import pytest

from longguard import GuardConfig
from longguard.integrations.crewai import (
    CrewGuard,
    GuardTerminatedException,
    add_guard_to_crew,
)


class MockCrewAction:
    def __init__(self, tool: str, tool_input: dict, thought: str = "Thinking", tokens: int = 100):
        self.tool = tool
        self.tool_input = tool_input
        self.thought = thought
        self.tokens_used = tokens


class MockCrewFinish:
    def __init__(self, output: str):
        self.output = output


def test_crewguard_step_callback_clean():
    """Test standard progression through steps without loops."""
    guard = CrewGuard(GuardConfig(tool_repeat_threshold=3))

    step1 = MockCrewAction(tool="search", tool_input={"q": "apple"})
    step2 = MockCrewAction(tool="calculator", tool_input={"expr": "1+1"})

    guard.step_callback(step1)
    guard.step_callback(step2)

    assert len(guard.report.steps) == 2
    assert guard.last_reflection is None


def test_crewguard_tool_repeat_trip():
    """Test that repeated identical tools trigger reflection then terminate."""
    guard = CrewGuard(GuardConfig(tool_repeat_threshold=2, max_reflections=1))

    step = MockCrewAction(tool="search", tool_input={"q": "same"})

    # Step 1: Clean
    guard.step_callback(step)
    assert guard.last_reflection is None

    # Step 2: Triggers reflection
    guard.step_callback(step)
    assert guard.last_reflection is not None

    # Step 3: Loop persists -> Trips circuit breaker
    with pytest.raises(GuardTerminatedException) as exc_info:
        guard.step_callback(step)

    assert "tool_repeat" in str(exc_info.value)
    assert exc_info.value.report is not None


def test_crewguard_dict_step_callback():
    """Test step_callback accepting dictionary step objects."""
    guard = CrewGuard(GuardConfig(tool_repeat_threshold=2, max_reflections=1))

    dict_step = {
        "thought": "Let me search",
        "tool": "web_search",
        "tool_input": {"url": "https://example.com"},
        "result": "content",
        "tokens": 50,
    }

    guard.step_callback(dict_step)
    guard.step_callback(dict_step)

    with pytest.raises(GuardTerminatedException):
        guard.step_callback(dict_step)


def test_crewguard_tuple_step_callback():
    """Test step_callback accepting (action, observation) tuples."""
    guard = CrewGuard(GuardConfig(tool_repeat_threshold=2, max_reflections=1))

    action = MockCrewAction(tool="db", tool_input={"sql": "SELECT 1"})
    tuple_step = (action, "row 1")

    guard.step_callback(tuple_step)
    guard.step_callback(tuple_step)

    with pytest.raises(GuardTerminatedException):
        guard.step_callback(tuple_step)


def test_crewguard_dollar_cost_cap():
    """Test CrewGuard tripping on max_cost_usd."""
    guard = CrewGuard(GuardConfig(model="gpt-4o", max_cost_usd=0.001))

    # Incur significant token consumption
    step = MockCrewAction(tool="analyze", tool_input={}, tokens=50_000)

    with pytest.raises(GuardTerminatedException) as exc_info:
        guard.step_callback(step)

    assert "cost_budget_exceeded" in str(exc_info.value)


def test_add_guard_to_crew():
    """Test add_guard_to_crew helper function attaching to crew and agents."""

    class MockAgent:
        def __init__(self, name: str):
            self.name = name
            self.step_callback = None

    class MockCrew:
        def __init__(self, agents):
            self.agents = agents
            self.step_callback = None

    agent1 = MockAgent("Agent 1")
    agent2 = MockAgent("Agent 2")
    crew = MockCrew(agents=[agent1, agent2])

    guarded_crew = add_guard_to_crew(crew, GuardConfig(tool_repeat_threshold=2))

    assert hasattr(guarded_crew, "__longguard__")
    guard = guarded_crew.__longguard__
    assert isinstance(guard, CrewGuard)

    # Check callbacks attached
    assert crew.step_callback == guard.step_callback
    assert agent1.step_callback == guard.step_callback
    assert agent2.step_callback == guard.step_callback

    # Trigger via agent callback
    step = MockCrewAction(tool="query", tool_input={})
    agent1.step_callback(step)
    assert len(guard.report.steps) == 1
    assert "Total Steps: 1" in guard.summary()


def test_add_guard_to_crew_chains_existing_callback():
    """Test that existing step_callbacks are preserved and called in chain."""
    existing_called = []

    def original_cb(step):
        existing_called.append(step)

    class MockCrew:
        def __init__(self):
            self.step_callback = original_cb
            self.agents = []

    crew = MockCrew()
    add_guard_to_crew(crew)

    step = MockCrewAction(tool="ping", tool_input={})
    crew.step_callback(step)

    assert len(existing_called) == 1
    assert existing_called[0] == step
