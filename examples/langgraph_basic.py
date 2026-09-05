"""
LongGuard + LangGraph: Basic Integration Example

This example demonstrates how to add LongGuard to a LangGraph
StateGraph with a single line of code. It simulates an agent
that gets stuck in a tool-repeat loop, and shows how LongGuard
detects the loop, injects a Reflect & Pivot prompt, and eventually
kills the agent when it doesn't recover.

Requirements:
    pip install longguard[langgraph]

Usage:
    python langgraph_basic.py
"""

from longguard import GuardConfig, CircuitBreaker, AgentStep


def simulate_looping_agent():
    """Simulate an agent that gets stuck calling the same tool repeatedly.

    This demonstrates the core LongGuard workflow without requiring
    a full LangGraph setup. Perfect for understanding how the circuit
    breaker works.
    """
    print("=" * 60)
    print("LongGuard Basic Example: Simulating a Looping Agent")
    print("=" * 60)
    print()

    # Create a config with strict thresholds for demo purposes
    config = GuardConfig(
        tool_repeat_threshold=3,
        tool_repeat_window=6,
        max_tokens_per_run=10_000,
        max_steps=20,
        max_reflections=2,
    )

    # Create the circuit breaker
    breaker = CircuitBreaker(config)

    # Simulate agent steps — first diverse, then stuck in a loop
    steps = [
        # Normal operation: different tools
        AgentStep(
            step_number=1,
            thought="I need to find the capital of France.",
            action="search",
            action_input="capital of France",
            observation="Paris is the capital of France.",
            tokens_used=120,
        ),
        AgentStep(
            step_number=2,
            thought="Now let me check the population.",
            action="lookup",
            action_input="population of Paris",
            observation="2.1 million people live in Paris.",
            tokens_used=150,
        ),
        # Stuck in a loop: same tool, same args
        AgentStep(
            step_number=3,
            thought="Let me search for the capital again.",
            action="search",
            action_input="capital of France",
            observation="Paris is the capital of France.",
            tokens_used=110,
        ),
        AgentStep(
            step_number=4,
            thought="The search didn't help. Let me try again.",
            action="search",
            action_input="capital of France",
            observation="Paris is the capital of France.",
            tokens_used=110,
        ),
        AgentStep(
            step_number=5,
            thought="Maybe if I search one more time...",
            action="search",
            action_input="capital of France",
            observation="Paris is the capital of France.",
            tokens_used=110,
        ),
        # If the agent continues the loop after reflection...
        AgentStep(
            step_number=6,
            thought="Still searching for the capital.",
            action="search",
            action_input="capital of France",
            observation="Paris is the capital of France.",
            tokens_used=110,
        ),
        AgentStep(
            step_number=7,
            thought="One more search attempt.",
            action="search",
            action_input="capital of France",
            observation="Paris is the capital of France.",
            tokens_used=110,
        ),
    ]

    # Run the agent through the circuit breaker
    for step in steps:
        decision = breaker.check(step)

        print(f"Step {step.step_number}: action={step.action}")
        print(f"  Decision: {decision.action}")

        if decision.action == "continue":
            print("  ✅ Proceeding normally")

        elif decision.action == "reflect":
            print(f"  ⚠️  Loop detected: {decision.reason}")
            print(f"  📝 Injecting Reflect & Pivot prompt:")
            # Print first 2 lines of the pivot prompt
            pivot_lines = decision.inject_prompt.split("\n")[:3]
            for line in pivot_lines:
                print(f"     {line}")
            print("     ...")

        elif decision.action == "kill":
            print(f"  🛑 Agent killed: {decision.reason}")
            break

        print()

    # Print the final report
    print("=" * 60)
    print("Guard Report")
    print("=" * 60)
    print(breaker.report.summary())
    print()
    print("Full report (JSON):")
    print(breaker.report.to_json(indent=2))


def simulate_recovery():
    """Simulate an agent that recovers after a Reflect & Pivot injection.

    This shows the happy path: the breaker detects a loop, injects
    a pivot prompt, and the agent changes approach, recovering to
    CLOSED state.
    """
    print("\n" + "=" * 60)
    print("LongGuard Recovery Example: Agent Pivots and Recovers")
    print("=" * 60)
    print()

    config = GuardConfig(
        tool_repeat_threshold=3,
        tool_repeat_window=6,
        max_tokens_per_run=10_000,
        max_steps=20,
        max_reflections=2,
    )

    breaker = CircuitBreaker(config)

    steps = [
        # Diverse steps
        AgentStep(
            step_number=1,
            thought="I need to find the weather.",
            action="search",
            action_input="weather in Tokyo",
            observation="Sunny, 22°C",
            tokens_used=100,
        ),
        # Loop begins
        AgentStep(
            step_number=2,
            thought="Let me search again.",
            action="search",
            action_input="weather in Tokyo",
            observation="Sunny, 22°C",
            tokens_used=100,
        ),
        AgentStep(
            step_number=3,
            thought="Search once more.",
            action="search",
            action_input="weather in Tokyo",
            observation="Sunny, 22°C",
            tokens_used=100,
        ),
        # After reflection: agent changes approach!
        AgentStep(
            step_number=4,
            thought="I already have the weather. Let me provide the answer.",
            action="calculator",
            action_input="22 * 9/5 + 32",
            observation="71.6°F",
            tokens_used=80,
        ),
        AgentStep(
            step_number=5,
            thought="The weather in Tokyo is sunny and 22°C (71.6°F).",
            action=None,
            observation=None,
            tokens_used=50,
        ),
    ]

    for step in steps:
        decision = breaker.check(step)
        status = "✅" if decision.action == "continue" else "⚠️" if decision.action == "reflect" else "🛑"
        print(f"Step {step.step_number}: {status} {decision.action} (state={breaker.state.value})")

    print()
    print(f"Final breaker state: {breaker.state.value}")
    print(f"Recovery successful: {breaker.state == BreakerState.CLOSED}")


if __name__ == "__main__":
    # Import BreakerState for the recovery example
    from longguard.core.breaker import BreakerState

    simulate_looping_agent()
    simulate_recovery()
