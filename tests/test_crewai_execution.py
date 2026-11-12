"""Real CrewAI sequential ReAct execution using a local model/tool boundary."""

# Imports follow optional dependency detection and credential isolation.
# ruff: noqa: E402

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.integration

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_STORAGE_DIR", "/tmp/longguard-crewai-tests")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
token_module = pytest.importorskip("crewai_core.token_manager")
credential_directory = TemporaryDirectory(prefix="longguard-test-credentials-")
with patch.object(
    token_module.TokenManager,
    "_get_secure_storage_path",
    return_value=Path(credential_directory.name),
):
    pytest.importorskip("crewai")

from crewai import Agent, BaseLLM, Crew, Task
from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.events.event_bus import crewai_event_bus
from crewai.tools import tool

from longguard import GuardConfig
from longguard.integrations.crewai import GuardTerminatedException, add_guard_to_crew


@pytest.fixture(autouse=True)
def isolate_framework_credentials():
    """Stub only external credential-file location; use actual framework execution."""
    with (
        patch.object(
            token_module.TokenManager,
            "_get_secure_storage_path",
            return_value=Path(credential_directory.name),
        ),
        crewai_event_bus.scoped_handlers(),
    ):
        yield


@pytest.fixture(scope="session", autouse=True)
def close_framework_event_loop():
    yield
    crewai_event_bus.shutdown(wait=False)


class LocalModel(BaseLLM):
    def __init__(self, *, recover, finish_first=False):
        super().__init__(model="local-fixture")
        self.calls = 0
        self.recover = recover
        self.finish_first = finish_first
        self.received_reflection = False

    def call(self, messages, **kwargs):
        self.calls += 1
        self.received_reflection |= any(
            "Reflect" in str(m) or "LONGGUARD" in str(m).upper() for m in messages
        )
        if self.finish_first or (self.recover and self.received_reflection):
            return "Thought: I will finish with available evidence.\nFinal Answer: finished"
        return 'Thought: Search again.\nAction: search\nAction Input: {"q": "same"}'

    def supports_function_calling(self):
        return False


def make_crew(*, recover=False, max_steps=30, finish_first=False):
    effects = []
    callbacks = []

    @tool
    def search(q: str) -> str:
        """Return a deterministic search result."""
        effects.append(q)
        return "same result"

    model = LocalModel(recover=recover, finish_first=finish_first)
    agent = Agent(
        role="Researcher",
        goal="Complete the task",
        backstory="Local fixture",
        llm=model,
        tools=[search],
        allow_delegation=False,
        max_iter=10,
        step_callback=lambda step: callbacks.append("agent"),
        executor_class=CrewAgentExecutor,
    )
    task = Task(description="Find the answer", expected_output="An answer", agent=agent)
    crew = Crew(
        agents=[agent],
        tasks=[task],
        cache=False,
        tracing=False,
        step_callback=lambda step: callbacks.append("crew"),
    )
    add_guard_to_crew(
        crew, GuardConfig(tool_repeat_threshold=2, max_reflections=1, max_steps=max_steps)
    )
    return crew, model, effects, callbacks


def test_real_crew_receives_reflection_and_recovers():
    crew, model, effects, callbacks = make_crew(recover=True)
    result = crew.kickoff()
    assert result.raw == "finished"
    assert model.received_reflection
    assert crew.__longguard__.report.reflections_injected == 1
    assert callbacks.count("agent") == callbacks.count("crew") == 5
    assert crew.__longguard__.report.total_steps == 3
    # CrewAI suppresses identical tool re-execution; repeated requests are
    # still observed, and their callbacks must not count twice per model turn.
    assert effects == ["same"]


def test_real_failed_recovery_stops_and_reattachment_is_idempotent():
    crew, model, effects, callbacks = make_crew()
    guard = crew.__longguard__
    assert add_guard_to_crew(crew).__longguard__ is guard
    with pytest.raises(GuardTerminatedException):
        crew.kickoff()
    assert model.calls == 3
    assert effects == ["same"]
    assert guard.report.total_steps == 3
    assert callbacks.count("agent") == callbacks.count("crew") == 6
    prior = guard.report
    with pytest.raises(GuardTerminatedException):
        crew.kickoff()
    assert guard.report.run_id != prior.run_id
    assert guard.report.total_steps == prior.total_steps == 3


def test_real_hard_cap_prevents_next_model_call_and_async_is_rejected():
    crew, model, effects, _ = make_crew(max_steps=1)
    with pytest.raises(GuardTerminatedException, match="max_steps"):
        crew.kickoff()
    assert model.calls == 2
    assert effects == ["same"]
    with pytest.raises(ValueError, match="synchronous"):
        crew.kickoff_async()


def test_real_normal_completion_has_no_unnecessary_intervention():
    crew, model, effects, callbacks = make_crew(finish_first=True)
    assert crew.kickoff().raw == "finished"
    assert model.calls == 1
    assert effects == []
    assert callbacks == ["crew", "agent"]
    assert crew.__longguard__.report.reflections_injected == 0
    assert crew.__longguard__.report.usage_warnings


def test_unsupported_mode_is_rejected_without_partial_attachment():
    crew, _, _, _ = make_crew(finish_first=True)
    # A separate real crew has the unsupported execution mode before attachment.
    fresh = Crew(
        agents=crew.agents,
        tasks=crew.tasks,
        process="hierarchical",
        manager_llm=LocalModel(recover=False),
        tracing=False,
    )
    prior = fresh.step_callback
    with pytest.raises(ValueError, match="sequential"):
        add_guard_to_crew(fresh)
    assert fresh.step_callback is prior
    assert not hasattr(fresh, "__longguard__")


def test_shared_guarded_agents_are_rejected_before_callbacks_are_changed():
    crew, _, _, _ = make_crew(finish_first=True)
    fresh = Crew(agents=crew.agents, tasks=crew.tasks, tracing=False)
    prior = fresh.step_callback
    with pytest.raises(ValueError, match="Sharing"):
        add_guard_to_crew(fresh)
    assert fresh.step_callback is prior
    assert not hasattr(fresh, "__longguard__")


@pytest.mark.parametrize(
    "owner,field,message",
    [
        ("task", "async_execution", "synchronous"),
        ("crew", "planning", "planning/memory"),
        ("crew", "memory", "planning/memory"),
        ("agent", "allow_delegation", "delegation/reasoning"),
        ("agent", "reasoning", "delegation/reasoning"),
    ],
)
def test_unsupported_mode_changes_fail_before_a_model_or_tool_starts(owner, field, message):
    crew, model, effects, _ = make_crew(finish_first=True)
    target = {"task": crew.tasks[0], "crew": crew, "agent": crew.agents[0]}[owner]
    setattr(target, field, True)
    with pytest.raises(ValueError, match=message):
        crew.kickoff()
    assert model.calls == 0
    assert effects == []
    assert crew.__longguard__.report.total_steps == 0


def test_native_tool_model_and_default_executor_require_a_supported_mode():
    from crewai.experimental import AgentExecutor

    crew, model, effects, _ = make_crew(finish_first=True)
    crew.agents[0].executor_class = AgentExecutor
    with pytest.raises(ValueError, match="CrewAgentExecutor"):
        crew.kickoff()
    crew.agents[0].executor_class = CrewAgentExecutor

    class NativeModel(LocalModel):
        def supports_function_calling(self):
            return True

    native = NativeModel(recover=False, finish_first=True)
    crew.agents[0].llm = native
    with pytest.raises(ValueError, match="native tool batches"):
        crew.kickoff()
    assert model.calls == native.calls == 0
    assert effects == []


def test_checkpoint_continuation_and_conflicting_reattachment_are_rejected():
    from longguard.integrations.crewai import CrewGuard

    crew, model, _, _ = make_crew(finish_first=True)
    with pytest.raises(ValueError, match="checkpoint"):
        crew.kickoff(from_checkpoint="saved-state")
    with pytest.raises(ValueError, match="different"):
        add_guard_to_crew(crew, GuardConfig(max_steps=1))
    other = Crew(agents=crew.agents, tasks=crew.tasks, tracing=False)
    with pytest.raises(ValueError, match="separate guard"):
        add_guard_to_crew(other, guard=crew.__longguard__)
    with pytest.raises(ValueError, match="different"):
        add_guard_to_crew(crew, guard=CrewGuard())
    assert model.calls == 0


def test_missing_public_hook_api_is_reported_before_execution(monkeypatch):
    import builtins

    crew, model, effects, _ = make_crew(finish_first=True)
    real_import = builtins.__import__

    def without_hook_api(name, *args, **kwargs):
        if name == "crewai.hooks.dispatch":
            raise ImportError("Legacy CrewAI lacks this API")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_hook_api)
    with pytest.raises(ValueError, match="execution-scoped hook API"):
        crew.kickoff()
    assert model.calls == 0
    assert effects == []


def test_concurrent_kickoff_cannot_reset_the_active_budget():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    entered, release = Event(), Event()
    crew, _, effects, _ = make_crew(finish_first=True)

    class BlockingModel(LocalModel):
        def call(self, messages, **kwargs):
            entered.set()
            assert release.wait(5)
            return super().call(messages, **kwargs)

    model = BlockingModel(recover=False, finish_first=True)
    crew.agents[0].llm = model
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(crew.kickoff)
        assert entered.wait(5)
        try:
            with pytest.raises(ValueError, match="Concurrent execution"):
                crew.kickoff()
        finally:
            release.set()
        assert future.result(timeout=5).raw == "finished"
    assert model.calls == 1
    assert effects == []
    assert crew.__longguard__.get_report().total_steps == 1
    assert "Total Steps: 1" in crew.__longguard__.summary()
