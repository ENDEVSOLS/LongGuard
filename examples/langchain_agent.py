"""
LongGuard + LangChain: AgentExecutor Middleware Example

This example demonstrates how to use GuardedAgentExecutor to wrap
a LangChain AgentExecutor with LongGuard loop detection.

Since setting up a full LangChain agent requires API keys and
model access, this example shows the integration pattern and
demonstrates the manual step-checking API.

Requirements:
    pip install longguard[langchain]

Usage:
    python langchain_agent.py
"""

from longguard import GuardConfig, CircuitBreaker, AgentStep
from longguard.core.breaker import BreakerState


def manual_step_checking():
    """Demonstrate manual step checking for custom agent loops.

    If you're building your own agent loop (not using AgentExecutor),
    you can call breaker.check() directly after each step. This gives
    you full control over how to handle each decision.
    """
    print("=" * 60)
    print("LongGuard Manual Step Checking Example")
    print("=" * 60)
    print()

    config = GuardConfig(
        tool_repeat_threshold=3,
        semantic_variance_threshold=0.2,
        dead_end_threshold=4,
        max_tokens_per_run=50_000,
        max_steps=25,
        max_reflections=2,
    )

    breaker = CircuitBreaker(config)

    # Simulate a custom agent loop
    class CustomAgent:
        """A simulated agent that makes tool calls."""

        def __init__(self):
            self.step_count = 0

        def run_step(self, query: str) -> AgentStep:
            """Simulate a single agent step."""
            self.step_count += 1

            # Simulate different behaviors based on step count
            if self.step_count <= 2:
                return AgentStep(
                    step_number=self.step_count,
                    thought=f"Processing query: {query}",
                    action="search",
                    action_input=query,
                    observation="Found relevant information.",
                    tokens_used=200,
                )
            else:
                # Agent gets stuck
                return AgentStep(
                    step_number=self.step_count,
                    thought="Let me search for more information.",
                    action="search",
                    action_input=query,
                    observation="No additional results.",
                    tokens_used=180,
                )

    agent = CustomAgent()
    query = "What is machine learning?"

    print(f"Running agent for query: '{query}'")
    print()

    while True:
        step = agent.run_step(query)
        decision = breaker.check(step)

        print(f"Step {step.step_number}: {step.action} → {decision.action}")

        if decision.action == "continue":
            # Normal operation — agent proceeds
            if step.observation:
                print(f"  Observation: {step.observation[:50]}...")

        elif decision.action == "reflect":
            # Loop detected — inject pivot prompt
            print(f"  ⚠️  Loop pattern: {decision.reason}")
            print(f"  Injecting Reflect & Pivot prompt to agent")
            # In a real agent, you would add the pivot prompt to the
            # agent's context/scratchpad here

        elif decision.action == "kill":
            # Agent is stuck beyond recovery
            print(f"  🛑 Agent terminated: {decision.reason}")
            break

        # Safety check
        if breaker.state == BreakerState.OPEN:
            break

        if agent.step_count >= 15:
            print("  Reached step limit for demo.")
            break

        print()

    # Print the report
    print()
    print("Run Report:")
    print(breaker.report.summary())


def guarded_agent_executor_pattern():
    """Show the pattern for using GuardedAgentExecutor with LangChain.

    This is a code pattern (not runnable without a real LangChain setup)
    showing how to integrate LongGuard with AgentExecutor.
    """
    print("\n" + "=" * 60)
    print("LongGuard + LangChain AgentExecutor Pattern")
    print("=" * 60)
    print()

    pattern = '''
    from langchain.agents import AgentExecutor
    from longguard.integrations.langchain import GuardedAgentExecutor
    from longguard import GuardConfig

    # Create your LangChain agent as usual
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
    )

    # Wrap with LongGuard
    config = GuardConfig(
        max_tokens_per_run=100_000,
        tool_repeat_threshold=3,
    )
    guarded = GuardedAgentExecutor(agent_executor, config)

    # Run — LongGuard monitors every step
    try:
        result = guarded.run("What is the weather in Tokyo?")
        print(result)
    except GuardTerminatedException as e:
        print(f"Agent was terminated: {e.reason}")
        print(f"Report: {e.report.summary()}")
    '''

    print(pattern)


if __name__ == "__main__":
    manual_step_checking()
    guarded_agent_executor_pattern()
