"""Legacy graph wrapper contracts through public nodes and detector extensions."""

import builtins
from types import SimpleNamespace

import pytest

from longguard import AbstractLoopDetector, CircuitBreakerTrippedError, DetectionResult, GuardConfig
from longguard.integrations.langgraph import LongGuard, add_guard_to_graph

pytestmark = pytest.mark.integration


class RecordingDetector(AbstractLoopDetector):
    """An application detector that records the observable step it receives."""

    def __init__(self):
        self.steps = []

    def analyze(self, step):
        self.steps.append(step)
        return DetectionResult()

    def reset(self):
        self.steps.clear()

    @property
    def name(self):
        return "RecordingDetector"


@pytest.mark.parametrize("objects", [False, True])
def test_node_receives_multimodal_text_and_complete_tool_observation(objects):
    guard = LongGuard()
    recorder = RecordingDetector()
    guard.breaker.detectors.append(recorder)
    messages = [
        {
            "type": "ai",
            "content": [{"text": "Read the row"}, "then compare"],
            "tool_calls": [{"id": "read-1", "name": "read", "args": {"row": 1}}],
        },
        {"type": "tool", "content": [{"text": "Found row"}, "A=4"], "tool_call_id": "read-1"},
    ]
    if objects:
        messages = [SimpleNamespace(**message) for message in messages]
    result = guard.wrap_node("agent", lambda state: {**state, "done": True})(
        {"messages": messages}, {"configurable": {"thread_id": "node"}}
    )
    assert result["done"]
    step = recorder.steps[0]
    assert step.thought == "Read the row\nthen compare"
    assert step.observation == "Found row\nA=4"
    assert step.action == "read" and step.action_input == {"row": 1}
    assert guard.get_report().total_steps == 1


@pytest.mark.parametrize(
    "arguments,expected", [(' {"row": 2}', {"row": 2}), ("incomplete JSON", "incomplete JSON")]
)
def test_original_provider_tool_arguments_survive_normalization(arguments, expected):
    guard = LongGuard()
    recorder = RecordingDetector()
    guard.breaker.detectors.append(recorder)
    state = {
        "messages": [
            {
                "content": "read",
                "additional_kwargs": {
                    "tool_calls": [
                        {"id": "read-2", "function": {"name": "read", "arguments": arguments}},
                    ]
                },
            }
        ]
    }
    guard.wrap_node("agent", lambda state: state)(state)
    assert recorder.steps[0].action == "read"
    assert recorder.steps[0].action_input == expected


def test_config_aware_node_and_internal_type_error_are_called_once():
    effects = []
    guard = LongGuard()

    def node(state, config):
        effects.append(config["request"])
        raise TypeError("Tool rejected its payload")

    with pytest.raises(TypeError, match="Tool rejected"):
        guard.wrap_node("agent", node)({"messages": []}, {"request": "one"})
    assert effects == ["one"]
    assert guard.get_report().total_steps == 1


def test_human_and_tool_only_histories_have_truthful_fallback_observations():
    guard = LongGuard()
    recorder = RecordingDetector()
    guard.breaker.detectors.append(recorder)
    node = guard.wrap_node("agent", lambda state: state)
    node({"messages": [SimpleNamespace(type="human", content="User request")]})
    assert recorder.steps[-1].thought == "User request"
    node({"messages": [{"type": "tool", "content": "", "tool_call_id": "previous"}]})
    assert recorder.steps[-1].thought == "(no thought)"
    assert recorder.steps[-1].observation == ""


def test_missing_optional_dependency_keeps_dict_reflection_and_termination_working(monkeypatch):
    real_import = builtins.__import__

    def without_langchain(name, *args, **kwargs):
        if name.startswith("langchain_core"):
            raise ImportError("Optional framework is absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_langchain)
    guard = LongGuard(GuardConfig(tool_repeat_threshold=1, max_reflections=1))
    effects = []
    node = guard.wrap_node("agent", lambda state: effects.append("agent") or state)
    state = {
        "messages": [
            {
                "type": "ai",
                "content": "search",
                "tool_calls": [
                    {"id": "call", "name": "search", "args": {}},
                ],
            },
            {"type": "tool", "content": "real result", "tool_call_id": "call"},
        ]
    }
    reflected = node(state)
    assert reflected["messages"][-1]["type"] == "system"
    assert "LONGGUARD" in reflected["messages"][-1]["content"]
    stopped = node(reflected)
    assert stopped["__longguard_terminated__"]
    assert stopped["messages"][-1]["type"] == "ai"
    assert "terminated" in stopped["messages"][-1]["content"]
    assert effects == ["agent"]


def test_legacy_nodes_and_excluded_dispatch_keep_accounting_separate():
    sentinel = object()
    graph = SimpleNamespace(
        _nodes={
            "__start__": sentinel,
            "__end__": sentinel,
            "agent": lambda state: state,
            "tools": lambda state: state,
        }
    )
    add_guard_to_graph(graph, GuardConfig(max_steps=1))
    assert graph._nodes["__start__"] is sentinel
    assert graph._nodes["__end__"] is sentinel
    graph._nodes["tools"]({"messages": []})
    assert graph.__longguard__.get_report().total_steps == 0
    graph._nodes["agent"]({"messages": []})
    stopped = graph._nodes["agent"]({"messages": []})
    with pytest.raises(CircuitBreakerTrippedError, match="max_steps"):
        graph._nodes["tools"](stopped)
    with pytest.raises(ValueError, match="different"):
        add_guard_to_graph(graph, GuardConfig(max_steps=2))


def test_invalid_graph_is_returned_with_an_explanatory_warning(caplog):
    graph = SimpleNamespace()
    assert add_guard_to_graph(graph) is graph
    assert "Could not find nodes dict" in caplog.text
