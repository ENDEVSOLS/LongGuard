<div align="center">

<p align="center">
  <img src="https://raw.githubusercontent.com/ENDEVSOLS/LongGuard/main/assets/logo.svg" alt="LongGuard Logo" width="460"/>
</p>

**In-flight circuit breaker & reasoning loop recovery for LangGraph & LangChain agents**

[![PyPI version](https://img.shields.io/pypi/v/longguard.svg?color=blue)](https://pypi.org/project/longguard/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/longguard?period=total&units=international_system&left_color=black&right_color=green&left_text=downloads)](https://pepy.tech/projects/longguard)
[![Python Versions](https://img.shields.io/pypi/pyversions/longguard.svg)](https://pypi.org/project/longguard/)
[![CI](https://github.com/ENDEVSOLS/LongGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/ENDEVSOLS/LongGuard/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://endevsols.github.io/LongGuard/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0%2B-orange.svg)](https://github.com/langchain-ai/langgraph)
[![LangChain](https://img.shields.io/badge/LangChain-1.0%2B-green.svg)](https://github.com/langchain-ai/langchain)

<p align="center">
  <a href="#quick-start"><strong>⚡ Quick Start</strong></a> &nbsp;·&nbsp;
  <a href="https://endevsols.github.io/LongGuard/"><strong>📖 Documentation</strong></a> &nbsp;·&nbsp;
  <a href="#why-longguard"><strong>Why LongGuard?</strong></a> &nbsp;·&nbsp;
  <a href="#part-of-the-long-suite"><strong>🌐 Part of Long Suite</strong></a> &nbsp;·&nbsp;
  <a href="#the-4-loop-detectors"><strong>Loop Detectors</strong></a> &nbsp;·&nbsp;
  <a href="#cicd-ready"><strong>CI/CD Ready</strong></a>
</p>

</div>

---

## Overview

> *"Why did my agent just burn $14 repeating the exact same search 20 times?"* — now you have an in-flight circuit breaker that detects loops and recovers gracefully.

When autonomous LLM agents hit an unexpected hurdle, they often get trapped in repetitive reasoning loops: calling identical tools with the same parameters, oscillating between two conflicting thoughts, or drifting aimlessly while burning thousands of tokens.

LangGraph's built-in `recursion_limit` is a **hard crash** (`GraphRecursionError`). It drops state, discards user context, and provides zero opportunity for recovery.

**LongGuard is an intelligent circuit breaker middleware.** It monitors your agent's chain-of-thought in real time, catches loops early, injects a **"Reflect & Pivot"** prompt to guide the agent back on track, and only halts (`kill`) gracefully if recovery fails—preserving complete state and token analytics.

---

## 🌐 Part of the Long Suite

LongGuard is part of the **[EnDevSols Long Suite](https://endevsols.com/open-source)** of open-source production AI tools:

- **[LongParser](https://github.com/ENDEVSOLS/LongParser)** — High-speed, privacy-first local document ingestion & chunking (PDF, DOCX, PPTX, XLSX)
- **[LongTrainer](https://github.com/ENDEVSOLS/Long-Trainer)** — Production multi-tenant RAG chatbot and agent framework
- **[LongTracer](https://github.com/ENDEVSOLS/LongTracer)** — Post-generation hallucination detection via hybrid STS + NLI claim verification
- **[LongProbe](https://github.com/ENDEVSOLS/LongProbe)** — Sub-second RAG retrieval regression testing with pytest
- **[LongGuard](https://github.com/ENDEVSOLS/LongGuard)** — **In-flight runtime agent circuit breaker & reasoning loop recovery** ← *You are here*

Together, the Long Suite covers the full AI lifecycle from data ingestion and retrieval CI regression to runtime agent safety and post-generation verification.

---

## 💡 Why LongGuard?

- ⚡ **Sub-millisecond overhead**: Evaluates in-flight agent steps without slowing down LLM inference.
- 🔄 **4 loop detectors**: Catches tool repetition, semantic oscillation, dead-end drift, and token velocity spikes.
- 🧭 **Reflect & Pivot prompt injection**: Guides stuck agents back on track before giving up.
- 🛡️ **Zero unhandled crashes**: Gracefully terminates and preserves conversation state if recovery fails.
- 🔌 **1-line integration**: Drop-in wrapper for LangGraph 1.0+ (`add_guard_to_graph`) and LangChain (`GuardedAgentExecutor`).
- 📊 **Full observability**: Generates detailed `GuardReport` summaries with per-step token tracking.
- 🧪 **100% test coverage**: 179 passing unit/integration tests, strict MyPy typing, and Ruff linted.

---

## 🏗️ Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/ENDEVSOLS/LongGuard/main/assets/architecture.svg" alt="LongGuard Architecture Flow" width="100%" />
</p>

---


## ⚡ Quick Start

### Installation

```bash
# Core package (standalone, zero heavy dependencies)
pip install longguard

# With LangGraph integration (LangGraph 1.0+)
pip install longguard[langgraph]

# With LangChain integration
pip install longguard[langchain]

# With high-quality sentence embeddings
pip install longguard[embeddings]

# Everything
pip install longguard[all]
```

### 1. LangGraph Integration (1 Line)

Compatible with **LangGraph 1.0+** and modern multimodal models (Claude 3.7, Gemini 2.5, GPT-4o):

```python
from langgraph.graph import StateGraph
from longguard.integrations.langgraph import add_guard_to_graph
from longguard import GuardConfig

# 1. Build your LangGraph workflow as usual
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_edge("agent", "tools")
workflow.add_conditional_edges("tools", should_continue)

# 2. Add LongGuard in one line!
workflow = add_guard_to_graph(workflow, GuardConfig())
app = workflow.compile()

# 3. Read execution telemetry after run
guard = workflow.__longguard__
print(guard.get_report().summary())
```

### 2. Standalone / Custom Agent Loop

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
        # Inject the recovery advice into your agent's context
        messages.append({"role": "user", "content": decision.inject_prompt})
    elif decision.action == "kill":
        print(f"Halted safely: {decision.reason}")
        break

# View summary report
print(breaker.report.summary())
```

---

## 🔍 The 4 Loop Detectors

| Detector | What It Catches | Real-World Example |
|---|---|---|
| **🔄 Tool Repeat** | Calling the same tool with identical inputs $\ge N$ times | Agent calls `search("revenue 2025")` 4 times with zero parameter changes |
| **🌀 Semantic Oscillation** | Cycling between the same concepts in reasoning | Agent reasons "Option A", then "No, B", then "Actually A", then "No, B" |
| **📉 Dead-End Drift** | Zero new information or observations for 5+ steps | Queries return empty results or repetitive error strings |
| **⚡ Token Velocity** | Sudden exponential token spikes per step | Agent injects giant raw HTML payloads into context, blowing budget |

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

All behavior is customizable through `GuardConfig`:

```python
from longguard import GuardConfig

config = GuardConfig(
    # Loop Detection Sensitivity
    tool_repeat_threshold=3,          # Repeated tool calls before reflection
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

## 🧪 CI/CD Ready

Run the test suite locally with `uv` or `pytest`:

```bash
# Run all 179 unit & integration tests
uv run pytest tests/ -v

# Run with coverage report
uv run pytest tests/ --cov=longguard --cov-report=term-missing

# Run code style & type checks
uv run ruff check src/ tests/
uv run mypy src/
```

Every push and pull request is automatically tested across **Python 3.10, 3.11, and 3.12** on both Ubuntu and macOS via GitHub Actions.

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on code formatting, running tests, and opening pull requests.

## 🛡️ Security

For vulnerability disclosures, please review [SECURITY.md](SECURITY.md) or contact security@endevsols.com.

## 📄 License

LongGuard is open-source software released under the [MIT License](LICENSE).
