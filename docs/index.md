# LongGuard 🛡️

**The Chain-of-Thought Circuit Breaker for LangGraph and LangChain Agents.**

[![PyPI version](https://img.shields.io/pypi/v/longguard.svg?color=blue)](https://pypi.org/project/longguard/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 179 passing](https://img.shields.io/badge/tests-179%20passing-brightgreen.svg)]()
[![Coverage: 90%](https://img.shields.io/badge/coverage-90%25-green.svg)]()

---

## What is LongGuard?

LongGuard is a lightweight runtime safety net for autonomous LLM agents.

When agents encounter ambiguous tool outputs or unexpected failures, they often get stuck:
- Calling the exact same tool with the exact same inputs 25 times in a row.
- Cycling between two reasoning steps in an infinite loop.
- Pursuing a dead-end direction that burns 30,000+ tokens.

Frameworks like LangGraph have a built-in `recursion_limit`, but it is a **hard crash**: it burns tokens until the limit, crashes with an unhandled `GraphRecursionError`, and loses the conversation state.

**LongGuard catches loops early (typically within 4 steps), injects a "Reflect & Pivot" prompt to help the agent course-correct, and only kills the run gracefully if recovery fails.**

```
CLOSED ──(loop detected)──▶ REFLECTING ──(persists)──▶ HALF_OPEN ──▶ OPEN (Kill)
   ▲                             │                         │
   └──────────(clean step)───────┴─────────────────────────┘
```

---

## Core Value Proposition

| What Happens Without LongGuard | What Happens With LongGuard |
|---|---|
| Agent repeats same tool 25 times | Trapped at **step 4** |
| Crashes with unhandled `500` error | **Graceful exit** with AI apology message |
| Burns 40k+ tokens ($2.00–$5.00+ / query) | **Hard spend cap** & early intervention |
| Agent never knows why it failed | Injected **Reflect & Pivot prompt** forces recovery |
| Zero post-run insight | Detailed **GuardReport** with step-by-step audit |

---

## 30-Second Example (LangGraph)

LongGuard wraps your agent nodes in a single line of code:

```python
from langgraph.graph import StateGraph
from longguard.integrations.langgraph import add_guard_to_graph
from longguard import GuardConfig

# 1. Build your StateGraph as usual
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge("agent", "tools")
...

# 2. Add LongGuard — one line!
workflow = add_guard_to_graph(workflow, GuardConfig())
app = workflow.compile()
```

---

## Next Steps

- Check out the [Quick Start Guide](getting-started.md) to install and configure LongGuard.
- Learn about the [4 Built-In Detectors](concepts/detectors.md).
- Explore [LangGraph Integration](integrations/langgraph.md) and [LangChain Integration](integrations/langchain.md).
