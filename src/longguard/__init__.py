"""LongGuard — Chain-of-Thought Circuit Breaker for LangGraph / LangChain Agents.

LongGuard monitors every step of an LLM agent execution, detects semantic
reasoning loops, and injects "Reflect & Pivot" prompts to recover — or
gracefully terminates the agent when recovery fails.

Quick start::

    from longguard import GuardConfig, CircuitBreaker, AgentStep

    config = GuardConfig(max_tokens_per_run=100_000)
    breaker = CircuitBreaker(config)

    for step in agent_loop:
        agent_step = AgentStep(
            step_number=i,
            thought=step.thought,
            action=step.tool,
            action_input=step.args,
            observation=step.result,
            tokens_used=step.tokens,
        )
        decision = breaker.check(agent_step)

        if decision.action == "kill":
            print(f"Agent killed: {decision.reason}")
            break
        elif decision.action == "reflect":
            print(f"Injecting pivot: {decision.inject_prompt}")
        # else: continue normally

LangGraph integration (one line)::

    from longguard.integrations.langgraph import add_guard_to_graph
    workflow = add_guard_to_graph(workflow, GuardConfig())

LangChain integration::

    from longguard.integrations.langchain import GuardedAgentExecutor
    guarded = GuardedAgentExecutor(agent_executor, GuardConfig())
    result = guarded.run("Your query here")
"""

__version__ = "0.1.1"

from typing import Any

from .config import GuardConfig
from .core.breaker import BreakerDecision, BreakerState, CircuitBreaker
from .core.detectors import (
    AbstractLoopDetector,
    DeadEndDriftDetector,
    DetectionResult,
    Embedder,
    HashBasedEmbedder,
    SemanticOscillationDetector,
    SentenceTransformerEmbedder,
    TokenVelocityDetector,
    ToolRepeatDetector,
)
from .core.pivot import PIVOT_TEMPLATES, ReflectAndPivotInjector
from .core.reporter import GuardReport
from .core.step import AgentStep

# --- Lazy integration imports ---
# These are optional and depend on langgraph / langchain being installed.
# Users who only use the core API should not need those packages.

def __getattr__(name: str) -> Any:
    """Lazy-load integration classes on first access."""
    _langgraph_names = {"LongGuard", "add_guard_to_graph"}
    _langchain_names = {"GuardedAgentExecutor", "GuardTerminatedException"}
    _strands_names = {"StrandsGuard", "GuardTerminatedError"}

    if name in _langgraph_names:
        from .integrations.langgraph import LongGuard, add_guard_to_graph
        return {"LongGuard": LongGuard, "add_guard_to_graph": add_guard_to_graph}[name]

    if name in _langchain_names:
        from .integrations.langchain import GuardedAgentExecutor, GuardTerminatedException
        return {
            "GuardedAgentExecutor": GuardedAgentExecutor,
            "GuardTerminatedException": GuardTerminatedException,
        }[name]

    if name in _strands_names:
        from .integrations.strands import GuardTerminatedError, StrandsGuard
        return {
            "StrandsGuard": StrandsGuard,
            "GuardTerminatedError": GuardTerminatedError,
        }[name]

    raise AttributeError(f"module 'longguard' has no attribute {name!r}")


__all__ = [
    # Config
    "GuardConfig",
    # Core models
    "AgentStep",
    # Detectors
    "AbstractLoopDetector",
    "DetectionResult",
    "Embedder",
    "ToolRepeatDetector",
    "SemanticOscillationDetector",
    "DeadEndDriftDetector",
    "TokenVelocityDetector",
    "HashBasedEmbedder",
    "SentenceTransformerEmbedder",
    # Circuit breaker
    "CircuitBreaker",
    "BreakerState",
    "BreakerDecision",
    # Pivot
    "ReflectAndPivotInjector",
    "PIVOT_TEMPLATES",
    # Reporting
    "GuardReport",
    # Integrations (lazy-loaded)
    "LongGuard",
    "add_guard_to_graph",
    "GuardedAgentExecutor",
    "GuardTerminatedException",
    "StrandsGuard",
    "GuardTerminatedError",
]
