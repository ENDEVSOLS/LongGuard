# LangGraph Integration

LongGuard provides seamless, non-invasive integration for **LangGraph 1.0+**.

---

## 1-Line StateGraph Wrapping

The easiest way to guard your LangGraph workflow is `add_guard_to_graph()`:

```python
from langgraph.graph import StateGraph, START, END
from longguard.integrations.langgraph import add_guard_to_graph
from longguard import GuardConfig

# 1. Build your StateGraph as usual
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")

# 2. Add LongGuard in one line
workflow = add_guard_to_graph(workflow, GuardConfig(
    tool_repeat_threshold=3,
    max_tokens_per_run=50_000,
    max_steps=25,
))

app = workflow.compile()
```

---

## What Happens When Terminated?

When the circuit breaker opens and kills the agent:
1. LongGuard preserves the full `messages` history.
2. It appends an explanation message to `messages`:
   ```
   [LongGuard] Agent terminated: reflection_failed: tool_repeat (confidence: 100%).
   The agent was stuck in a reasoning loop and could not recover after reflection attempts.
   ```
3. It sets `__longguard_terminated__ = True` and `__longguard_reason__ = "..."` on the state dictionary.
4. Your `should_continue` conditional edge can safely terminate to `END` without unhandled crashes:

```python
def should_continue(state: AgentState):
    if state.get("__longguard_terminated__"):
        return END
    if not state["messages"][-1].tool_calls:
        return END
    return "tools"
```

---

## Excluding Execution Nodes

By default, `add_guard_to_graph` ignores execution-only nodes (`"tools"`, `"action"`, `"__start__"`, `"__end__"`). You can customize this list:

```python
workflow = add_guard_to_graph(
    workflow,
    GuardConfig(),
    exclude_nodes=["tools", "my_custom_db_writer"],
)
```

---

## Accessing Telemetry

After execution, the guard report is stored on the graph instance:

```python
guard = workflow.__longguard__
print(guard.get_report().summary())
```
