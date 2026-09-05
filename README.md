<div align="center">

# LongGuard 🛡️

**In-Flight Circuit Breaker & Reasoning Loop Recovery for AI Agents**

*Stop runaway agent loops, prevent token budget blowouts, and inject "Reflect & Pivot" guidance before crashes happen.*

[![CI](https://github.com/ENDEVSOLS/LongGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/ENDEVSOLS/LongGuard/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs--material-blue.svg)](https://endevsols.github.io/LongGuard/)
[![PyPI version](https://img.shields.io/pypi/v/longguard.svg?color=blue)](https://pypi.org/project/longguard/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0%2B-orange.svg)](https://github.com/langchain-ai/langgraph)
[![LangChain](https://img.shields.io/badge/LangChain-1.0%2B-green.svg)](https://github.com/langchain-ai/langchain)

[**Read Documentation**](https://endevsols.github.io/LongGuard/) • [**Report Bug**](https://github.com/ENDEVSOLS/LongGuard/issues) • [**EnDevSols AI Suite**](https://github.com/ENDEVSOLS)

</div>

---

## ⚡ What is LongGuard?

When autonomous LLM agents (LangGraph, LangChain, or custom loops) hit an unexpected hurdle, they often **get stuck in repetitive reasoning loops**:
- Calling the exact same tool with the exact same parameters over and over.
- Spinning between two reasoning thoughts (semantic oscillation).
- Aimlessly wandering with zero progress while burning thousands of tokens.

Frameworks like LangGraph have a built-in `recursion_limit`, but it is a **hard crash** (`GraphRecursionError`) that drops state, fails the user request, and provides zero opportunity for recovery.

**LongGuard is an intelligent circuit breaker middleware.** It monitors your agent's chain-of-thought in real time, catches loops early, injects a **"Reflect & Pivot"** prompt to guide the agent back on track, and only terminates gracefully if recovery fails.

```
                  ┌─────────────────────────────────┐
                  │      Agent Execution Loop       │
                  └────────────────┬────────────────┘
                                   │  Step N
                                   ▼
                        ┌─────────────────────┐
                        │   LongGuard Hook    │
                        └──────────┬──────────┘
                                   │
               ┌───────────────────┴───────────────────┐
               ▼                                       ▼
     [ No Loop Detected ]                     [ Loop Detected! ]
               │                                       │
        State: CLOSED                                  ▼
      (Normal execution)                      State: REFLECTING
                                                       │
                                            Inject "Reflect & Pivot"
                                            Prompt into Context
                                                       │
                                         ┌─────────────┴─────────────┐
                                         ▼                           ▼
                                    [ Recovers ]              [ Still Stuck ]
                                         │                           │
                                   State: CLOSED               State: OPEN
                                   (Runs to end)            (Graceful Termination)
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Core package (zero heavy dependencies)
pip install longguard

# With LangGraph integration
pip install longguard[langgraph]

# With LangChain integration
pip install longguard[langchain]
```

### 2. LangGraph Integration (1 Line)

Compatible with **LangGraph 1.0+** and modern multimodal models (Claude, Gemini, OpenAI):

```python
from langgraph.graph import StateGraph
from longguard.integrations.langgraph import add_guard_to_graph
from longguard import GuardConfig

# 1. Define your standard LangGraph workflow
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_edge("agent", "tools")
workflow.add_conditional_edges("tools", should_continue)

# 2. Wrap with LongGuard — that's it!
workflow = add_guard_to_graph(workflow, GuardConfig())
app = workflow.compile()

# 3. Access execution analytics after the run
guard = workflow.__longguard__
print(guard.get_report().summary())
```

### 3. Standalone / Custom Agent Loop

If you run a custom `while` loop or proprietary agent orchestrator:

```python
from longguard import CircuitBreaker, GuardConfig, AgentStep

breaker = CircuitBreaker(GuardConfig(
    tool_repeat_threshold=3,    # 3 identical tool calls = trigger
    max_tokens_per_run=50_000,  # Hard token cap
))

for step in run_agent():
    decision = breaker.check(AgentStep(
        step_number=step.index,
        thought=step.thought,
        action=step.tool_name,
        action_input=step.arguments,
        observation=step.tool_output,
        tokens_used=step.tokens,
    ))

    if decision.action == "reflect":
        # Inject the recovery advice into your agent's message list
        messages.append({"role": "user", "content": decision.inject_prompt})
    elif decision.action == "kill":
        print(f"Halted safely: {decision.reason}")
        break

# View summary report
print(breaker.report.summary())
```

---

## 🔍 The 4 Loop Detectors

LongGuard runs four lightweight detectors concurrently at every step:

| Detector | What It Catches | Real-World Example |
|---|---|---|
| **Tool Repeat** | Calling the same tool with identical inputs $\ge N$ times | Agent calls `web_search("apple stock 2026")` 4 times with zero param changes |
| **Semantic Oscillation** | Cycling between the same concepts in thoughts | Agent reasons "I should search A", then "No, B", then "Actually A", then "No, B" |
| **Dead-End Drift** | Zero new information or observations for 5+ steps | Agent makes queries that return blank results or repetitive error strings |
| **Token Velocity** | Sudden exponential token spikes per step | Agent dumps huge raw HTML payloads into thought context, blowing budget |

---

## 🔄 The Circuit Breaker State Machine

LongGuard adapts standard distributed systems circuit breaker patterns to LLM cognition:

1. **`CLOSED` (Normal)**: All checks pass. The agent runs freely.
2. **`REFLECTING` (Intervention)**: A loop was detected. LongGuard injects an automated **Reflect & Pivot prompt** instructing the agent:
   > *"You have called {tool} {count} times with the same arguments. Stop. Try a different tool or synthesize your current findings."*
3. **`HALF-OPEN` (Observation)**: The agent gets one chance to demonstrate progress following reflection.
4. **`OPEN` (Graceful Termination)**: If the loop persists after reflection, LongGuard terminates execution cleanly, preserving full trace telemetry and token usage.

---

## 📊 LongGuard vs. LangGraph `recursion_limit`

| Capability | LangGraph `recursion_limit` | LongGuard 🛡️ |
|---|:---:|:---:|
| **Detects Tool-Repeat Loops** | ❌ No | ✅ **Yes** |
| **Detects Semantic Reasoning Loops** | ❌ No | ✅ **Yes** |
| **Detects Sudden Cost / Token Spikes** | ❌ No | ✅ **Yes** |
| **Auto-Injects Recovery Prompts** | ❌ No | ✅ **Yes** |
| **Exit Behavior** | 💥 Unhandled Exception (`Crash`) | 🛡️ **Graceful State Preservation** |
| **Run Reporting & Telemetry** | ❌ No | ✅ **JSON & Summary Reports** |
| **Configurable Thresholds** | ❌ Single integer | ✅ **Granular `GuardConfig`** |

---

## ⚙️ Configuration at a Glance

All behavior can be customized via `GuardConfig`:

```python
from longguard import GuardConfig

config = GuardConfig(
    # Loop Detection Sensitivity
    tool_repeat_threshold=3,          # Number of repeated tool calls before reflection
    tool_repeat_window=6,             # History window to examine
    dead_end_threshold=5,             # Steps with no progress before triggering
    token_velocity_multiplier=3.0,    # Spike multiplier vs rolling baseline

    # Hard Safety Guardrails
    max_tokens_per_run=50_000,        # Hard stop if agent burns > 50k tokens
    max_steps=30,                     # Maximum steps permitted
    max_reflections=2,                # Maximum recovery attempts before kill
)
```

👉 *For detailed documentation on custom detectors, embedding backends, and LangSmith telemetry, see the [Full Documentation](https://endevsols.github.io/LongGuard/).*

---

## 🌐 Part of the EnDevSols AI Infrastructure Suite

LongGuard works alongside our other open-source tools to secure production LLM pipelines:

- [**LongParser**](https://github.com/ENDEVSOLS/LongParser) — Fast, privacy-first local document parser (PDF, DOCX, XLSX).
- [**LongProbe**](https://github.com/ENDEVSOLS/LongProbe) — Sub-second RAG retrieval regression testing with pytest.
- [**LongTracer**](https://github.com/ENDEVSOLS/LongTracer) — Post-generation hallucination detection & claim verification.
- [**LongGuard**](https://github.com/ENDEVSOLS/LongGuard) — In-flight runtime cognitive circuit breaker & loop recovery.

---

## 🤝 Contributing & Community

We love contributions!
- Submit bug reports and feature ideas via [GitHub Issues](https://github.com/ENDEVSOLS/LongGuard/issues).
- See [CONTRIBUTING.md](CONTRIBUTING.md) for development environment setup and testing guidelines.

## 📄 License

LongGuard is open-source software released under the [MIT License](LICENSE).
