# Quick Start Guide

This guide walks you through installing and using LongGuard in under two minutes.

---

## Installation

LongGuard is available on PyPI with optional dependency extras:

```bash
# Core package (zero heavy dependencies — works standalone)
pip install longguard

# With LangGraph support (LangGraph 1.0+)
pip install "longguard[langgraph]"

# With LangChain support
pip install "longguard[langchain]"

# With high-quality sentence embeddings (optional)
pip install "longguard[embeddings]"

# Everything included
pip install "longguard[all]"
```

Using [`uv`](https://github.com/astral-sh/uv):

```bash
uv add longguard
```

---

## 1. LangGraph Integration (1 Line)

Compatible with **LangGraph 1.0+** and modern multimodal models (Claude 3.5, GPT-4o, Gemini 1.5/2.0):

```python
from langgraph.graph import StateGraph
from longguard.integrations.langgraph import add_guard_to_graph
from longguard import GuardConfig

# Define your workflow
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_fn)
workflow.add_node("tools", tool_fn)
workflow.add_edge("agent", "tools")
workflow.add_conditional_edges("tools", should_continue)

# Instrument agent nodes in one line:
workflow = add_guard_to_graph(workflow, GuardConfig(
    tool_repeat_threshold=3,    # 3 identical calls triggers reflection
    max_tokens_per_run=50_000,  # Hard cap per query
    max_steps=25,               # Hard step cap
))

app = workflow.compile()

# Access telemetry report after execution:
guard = workflow.__longguard__
print(guard.get_report().summary())
```

> **Node Exclusion:** By default, `add_guard_to_graph` automatically instruments only agent reasoning nodes and skips execution-only nodes (`"tools"`, `"action"`, `"__start__"`, `"__end__"`).

---

## 2. Standalone / Custom Agent Loop

If you run a custom `while` loop or framework-agnostic pipeline:

```python
from longguard import CircuitBreaker, GuardConfig, AgentStep

breaker = CircuitBreaker(GuardConfig(
    tool_repeat_threshold=3,
    max_tokens_per_run=30_000,
))

for i, step in enumerate(agent_loop):
    agent_step = AgentStep(
        step_number=i + 1,
        thought=step.thought,
        action=step.tool_name,
        action_input=step.tool_args,
        observation=step.tool_result,
        tokens_used=step.tokens,
    )

    decision = breaker.check(agent_step)

    if decision.action == "continue":
        pass  # Proceed normally
    elif decision.action == "reflect":
        # Inject recovery prompt into context
        agent_context.append({"role": "system", "content": decision.inject_prompt})
    elif decision.action == "kill":
        print(f"Breaker opened! Reason: {decision.reason}")
        break

# View run summary
print(breaker.report.summary())
```

---

## 3. LangChain Integration

Wrap standard LangChain `AgentExecutor`:

```python
from langchain.agents import AgentExecutor
from longguard.integrations.langchain import GuardedAgentExecutor, GuardTerminatedException
from longguard import GuardConfig

executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools)
guarded = GuardedAgentExecutor(executor, GuardConfig())

try:
    result = guarded.run("Find information about quarterly revenue")
except GuardTerminatedException as exc:
    print(f"Terminated safely: {exc.reason}")
    print(exc.report.summary())
```
