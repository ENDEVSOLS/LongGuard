"""Existing callback-only CrewGuard accepts documented observable step shapes."""

from types import SimpleNamespace

import pytest

from longguard import AgentStep
from longguard.integrations.crewai import CrewGuard

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "event,action,tokens",
    [
        (AgentStep(4, "read", action="read", tokens_used=9), "read", 9),
        (
            SimpleNamespace(
                tool="read", tool_input={"row": 1}, log="Read the row", result="A=4", tokens_used=9
            ),
            "read",
            9,
        ),
        (SimpleNamespace(output="A=4", text="Use the result", tokens_used=9), None, 9),
        (
            (SimpleNamespace(tool="read", tool_input={"row": 1}, log="Read the row"), "A=4"),
            "read",
            0,
        ),
    ],
)
def test_callback_steps_preserve_action_usage_and_missing_usage_diagnostics(event, action, tokens):
    guard = CrewGuard()
    guard.step_callback(event)
    report = guard.get_report()
    assert report.total_steps == 1
    assert report.total_tokens == tokens
    assert report.steps[0]["action"] == action
    assert bool(report.usage_warnings) is (tokens == 0)


def test_callback_reset_and_summary_describe_a_new_run():
    guard = CrewGuard()
    guard.step_callback(AgentStep(1, "completed", tokens_used=9))
    previous = guard.get_report()
    assert "Total Steps: 1" in guard.summary()
    guard.reset()
    assert guard.get_report().run_id != previous.run_id
    assert guard.get_report().total_tokens == 0
    assert previous.total_tokens == 9
