"""Real LangGraph checkpoint, dispatch, isolation, and explicit resume contracts."""

# ruff: noqa: E402

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from longguard import GuardConfig
from longguard.integrations.langgraph import compile_guarded_graph


def build_graph(*, max_steps=30, checkpointer=None, recover=False):
    effects = []

    def agent(state):
        if any(
            isinstance(m, HumanMessage) and m.content == "finish" for m in state["messages"]
        ) or (recover and any(isinstance(m, SystemMessage) for m in state["messages"])):
            return {
                "messages": [
                    AIMessage(
                        content="finished",
                        usage_metadata={
                            "input_tokens": 2,
                            "output_tokens": 2,
                            "total_tokens": 4,
                        },
                    )
                ]
            }
        number = sum(isinstance(m, AIMessage) for m in state["messages"])
        return {
            "messages": [
                AIMessage(
                    content="search",
                    tool_calls=[
                        {"name": "search", "args": {"q": "same"}, "id": f"call_{number}"},
                    ],
                    usage_metadata={"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
                )
            ]
        }

    def tools(state):
        results = []
        for call in state["messages"][-1].tool_calls:
            effects.append(call["id"])
            results.append(ToolMessage(content="same result", tool_call_id=call["id"]))
        return {"messages": results}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        lambda state: "tools" if state["messages"][-1].tool_calls else END,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")
    runner = compile_guarded_graph(
        graph,
        GuardConfig(
            tool_repeat_threshold=2,
            max_reflections=1,
            max_steps=max_steps,
        ),
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
    )
    return runner, effects


def test_trip_cleans_calls_and_explicit_resume_retains_work_and_accounting():
    runner, effects = build_graph()
    config = {"configurable": {"thread_id": "one"}}
    stopped = runner.invoke({"messages": [HumanMessage(content="start")]}, config)
    assert stopped["__longguard_terminated__"] is True
    assert effects == ["call_0", "call_1"]
    messages = stopped["messages"]
    assert any(
        isinstance(m, ToolMessage) and m.tool_call_id == "call_2" and m.status == "error"
        for m in messages
    )
    report = runner.get_report("one")
    assert report.total_steps == 3
    assert report.total_tokens == 12
    assert report.tool_dispositions[0]["status"] == "skipped"
    with pytest.raises(ValueError, match="explicit"):
        runner.invoke(None, config)
    finished = runner.resume(config, correction="finish")
    assert finished["messages"][-1].content == "finished"
    assert effects == ["call_0", "call_1"]
    assert report.total_steps == 4
    assert report.total_tokens == 16
    assert report.continuations[0]["total_tokens"] == 12
    with pytest.raises(ValueError, match="stopped"):
        runner.resume(config, correction="finish")


def test_exhausted_allowance_and_unknown_thread_are_rejected():
    runner, effects = build_graph(max_steps=1)
    config = {"configurable": {"thread_id": "budget"}}
    stopped = runner.invoke({"messages": [HumanMessage(content="start")]}, config)
    assert stopped["__longguard_terminated__"]
    assert effects == ["call_0"]
    with pytest.raises(ValueError, match="allowance|hard"):
        runner.resume(config, correction="finish")
    with pytest.raises(ValueError, match="Unknown"):
        runner.resume({"configurable": {"thread_id": "missing"}}, correction="finish")


def test_independent_threads_and_normal_completion():
    runner, effects = build_graph()
    runner.invoke(
        {"messages": [HumanMessage(content="start")]}, {"configurable": {"thread_id": "a"}}
    )
    done = runner.invoke(
        {"messages": [HumanMessage(content="finish")]},
        {
            "configurable": {"thread_id": "b"},
        },
    )
    assert not done.get("__longguard_terminated__", False)
    assert runner.get_report("b").total_steps == 1
    assert runner.get_report("a").total_steps == 3
    assert runner.get_report("b").run_id != runner.get_report("a").run_id
    assert effects == ["call_0", "call_1"]


def test_concurrent_resume_does_not_duplicate_completed_actions():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    entered, release = Event(), Event()
    saver = InMemorySaver()
    runner, effects = build_graph(checkpointer=saver)
    config = {"configurable": {"thread_id": "concurrent"}}
    runner.invoke({"messages": [HumanMessage(content="start")]}, config)

    # A checkpointer's external write is the controllable system boundary.
    original_put = saver.put

    def delayed_put(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original_put(*args, **kwargs)

    saver.put = delayed_put
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(runner.resume, config, correction="finish")
        assert entered.wait(5)
        try:
            with pytest.raises(ValueError, match="already in progress"):
                runner.resume(config, correction="finish")
        finally:
            release.set()
        assert future.result(timeout=5)["messages"][-1].content == "finished"
    assert effects == ["call_0", "call_1"]


def test_legacy_excluded_tools_are_still_blocked_after_trip():
    from longguard import CircuitBreakerTrippedError
    from longguard.integrations.langgraph import add_guard_to_graph

    effects = []
    graph = StateGraph(MessagesState)
    graph.add_node("agent", lambda state: {"messages": []})
    graph.add_node("tools", lambda state: effects.append("dispatched") or {"messages": []})
    graph.add_edge(START, "agent")
    graph.add_edge("agent", "tools")
    graph.add_edge("tools", END)
    add_guard_to_graph(graph, GuardConfig(max_tokens_per_run=1))
    compiled = graph.compile()
    with pytest.raises(CircuitBreakerTrippedError):
        compiled.invoke({"messages": [AIMessage(content="x" * 100)]})
    assert effects == []


def test_normal_and_reflected_graph_completion():
    runner, effects = build_graph(recover=True)
    config = {"configurable": {"thread_id": "recovered"}}
    result = runner.invoke({"messages": [HumanMessage(content="start")]}, config)
    assert result["messages"][-1].content == "finished"
    assert effects == ["call_0", "call_1"]
    assert runner.get_report("recovered").reflections_injected == 1
    assert runner.get_report("recovered").total_tokens == 12
    assert runner.get_report("recovered").kill_reason is None
    other = {"configurable": {"thread_id": "normal"}}
    runner.invoke({"messages": [HumanMessage(content="finish")]}, other)
    assert runner.get_report("normal").total_steps == 1
    assert runner.get_report("normal").reflections_injected == 0


def test_nonmatching_checkpoint_and_legacy_reattachment():
    from longguard.integrations.langgraph import add_guard_to_graph

    saver = InMemorySaver()
    runner, effects = build_graph(checkpointer=saver)
    config = {"configurable": {"thread_id": "changed"}}
    runner.invoke({"messages": [HumanMessage(content="start")]}, config)
    snapshot = runner.get_state(config)
    checkpoint = saver.get(snapshot.config)
    # Simulate an external writer at the actual persistence boundary.
    checkpoint = {**checkpoint, "id": "ffffffff-ffff-7fff-bfff-ffffffffffff"}
    saver.put(snapshot.config, checkpoint, snapshot.metadata, {})
    with pytest.raises(ValueError, match="Checkpoint changed"):
        runner.resume(config, correction="finish")
    assert effects == ["call_0", "call_1"]
    graph = StateGraph(MessagesState)
    graph.add_node("agent", lambda state: {"messages": []})
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    add_guard_to_graph(graph)
    guard = graph.__longguard__
    add_guard_to_graph(graph)
    assert graph.__longguard__ is guard


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
def test_provider_cleanup_passes_real_message_parser_and_preserves_failed_result(protocol):
    from langchain_core.messages import convert_to_openai_messages

    from longguard.integrations.messages import resolve_pending_tool_calls

    if protocol == "anthropic":
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "failed", "name": "read", "input": {}},
                    {"type": "tool_use", "id": "pending", "name": "write", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "failed",
                        "content": "real failure",
                        "is_error": True,
                    },
                ],
            },
        ]
    else:
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": call_id,
                        "function": {"name": "read", "arguments": "{}"},
                    }
                    for call_id in ("failed", "pending")
                ],
            },
            {"role": "tool", "tool_call_id": "failed", "content": "real failure"},
        ]
    cleanup = resolve_pending_tool_calls(messages, "stop", protocol=protocol)
    parsed = convert_to_openai_messages(cleanup.messages)
    requested = [call["id"] for call in parsed[0]["tool_calls"]]
    answered = [message["tool_call_id"] for message in parsed[1:]]
    assert requested == answered == ["failed", "pending"]
    assert parsed[1]["content"] == "real failure"
    assert "not executed" in parsed[2]["content"]
    assert [item["tool_call_id"] for item in cleanup.dispositions] == ["pending"]


def test_controlled_runner_rejects_double_wrapped_graph():
    from longguard.integrations.langgraph import add_guard_to_graph

    graph = StateGraph(MessagesState)
    graph.add_node("agent", lambda state: {"messages": []})
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    add_guard_to_graph(graph)
    with pytest.raises(ValueError, match="unwrapped"):
        compile_guarded_graph(graph, checkpointer=InMemorySaver())


def minimal_graph(agent=None, *, start="agent", extra_node=False):
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent or (lambda state: {"messages": [AIMessage(content="done")]}))
    graph.add_node("tools", lambda state: {"messages": []})
    graph.add_edge(START, start)
    graph.add_conditional_edges("agent", lambda state: END, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    if extra_node:
        graph.add_node("extra", lambda state: {"messages": []})
        graph.add_edge(START, "extra")
        graph.add_edge("extra", END)
    return graph


@pytest.mark.parametrize(
    "settings,message",
    [
        ({"checkpointer": None}, "checkpointer"),
        ({"start": "tools"}, "topology"),
        ({"extra_node": True}, "only agent and tools"),
    ],
)
def test_controller_rejects_unproven_persistence_and_topologies(settings, message):
    saver = settings.get("checkpointer", InMemorySaver())
    graph = minimal_graph(
        start=settings.get("start", "agent"), extra_node=settings.get("extra_node", False)
    )
    with pytest.raises(ValueError, match=message):
        compile_guarded_graph(graph, checkpointer=saver)


@pytest.mark.parametrize(
    "config,message",
    [
        ({}, "thread_id"),
        ({"configurable": {"thread_id": ""}}, "thread_id"),
        ({"configurable": {"thread_id": 123}}, "thread_id"),
        ({"configurable": {"thread_id": "one", "checkpoint_ns": "branch"}}, "branching"),
        ({"configurable": {"thread_id": "one", "checkpoint_id": "old"}}, "branching"),
    ],
)
def test_invalid_thread_identifiers_and_checkpoint_branches_are_rejected(config, message):
    runner, effects = build_graph()
    with pytest.raises(ValueError, match=message):
        runner.invoke({"messages": [HumanMessage(content="start")]}, config)
    with pytest.raises(ValueError, match=message):
        runner.get_state(config)
    assert effects == []


def test_missing_input_and_unknown_report_do_not_create_a_run():
    runner, effects = build_graph()
    config = {"configurable": {"thread_id": "empty"}}
    with pytest.raises(ValueError, match="messages list"):
        runner.invoke(None, config)
    with pytest.raises(ValueError, match="Unknown thread"):
        runner.get_report("empty")
    assert effects == []


def test_existing_checkpoint_cannot_restart_without_matching_live_guard():
    saver = InMemorySaver()
    first, _ = build_graph(checkpointer=saver)
    config = {"configurable": {"thread_id": "existing"}}
    first.invoke({"messages": [HumanMessage(content="finish")]}, config)
    other, effects = build_graph(checkpointer=saver)
    with pytest.raises(ValueError, match="no matching live guard"):
        other.invoke({"messages": [HumanMessage(content="start")]}, config)
    assert effects == []


def test_missing_provider_usage_is_reported_as_an_estimate():
    runner = compile_guarded_graph(minimal_graph(), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "estimated"}}
    result = runner.invoke({"messages": [HumanMessage(content="finish")]}, config)
    assert result["messages"][-1].content == "done"
    assert runner.get_report("estimated").total_tokens == 1
    assert "estimated" in runner.get_report("estimated").usage_warnings[0]


def test_node_failure_is_not_misrepresented_as_a_resumable_guard_trip():
    effects = []

    def failing_agent(state):
        effects.append("attempt")
        raise RuntimeError("Local model unavailable")

    runner = compile_guarded_graph(minimal_graph(failing_agent), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "failed"}}
    with pytest.raises(RuntimeError, match="Local model unavailable"):
        runner.invoke({"messages": [HumanMessage(content="start")]}, config)
    with pytest.raises(ValueError, match="stopped checkpoint"):
        runner.resume(config, correction="retry")
    assert effects == ["attempt"]
    assert runner.get_report("failed").kill_reason is None
