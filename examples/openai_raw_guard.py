"""OpenAI Raw-Client Guard — LongGuard without LangGraph/LangChain.

This example shows how to use LongGuard directly in a plain ``while`` loop
that calls the OpenAI API — no LangGraph or LangChain required.

Usage
-----
Install dependencies::

    pip install longguard openai

Set your API key::

    export OPENAI_API_KEY=sk-...

Run::

    python examples/openai_raw_guard.py
"""

from __future__ import annotations

import time

# NOTE: Replace with a real OpenAI client call in production.
# This example uses a mock to run without API credentials.


def _mock_openai_client() -> object:
    """Return a minimal mock that mimics the OpenAI SDK for demo purposes."""
    import json
    from types import SimpleNamespace

    call_count = 0

    def create(**kwargs: object) -> object:  # noqa: ARG001
        nonlocal call_count
        call_count += 1

        # Simulate the same web_search loop 5 times to trigger detection
        if call_count <= 5:
            fn = SimpleNamespace(name="web_search", arguments=json.dumps({"query": "Python asyncio tutorial"}))
            tool_call = SimpleNamespace(function=fn)
            message = SimpleNamespace(content=None, tool_calls=[tool_call])
        else:
            message = SimpleNamespace(
                content="Based on my research, asyncio is Python's async framework.",
                tool_calls=None,
            )
        choice = SimpleNamespace(message=message)
        usage = SimpleNamespace(total_tokens=450 + call_count * 20)
        return SimpleNamespace(choices=[choice], usage=usage)

    completions = SimpleNamespace(create=create)
    chat = SimpleNamespace(completions=completions)
    return SimpleNamespace(chat=chat)


def main() -> None:
    # ── 1. Set up LongGuard ────────────────────────────────────────────────
    from longguard import AgentStep, CircuitBreaker, GuardConfig

    config = GuardConfig(
        model="gpt-4o",          # enables dollar-cost tracking
        max_cost_usd=0.50,        # hard kill if run exceeds $0.50
        max_steps=30,             # hard kill after 30 steps
        tool_repeat_threshold=3,  # detect repeating the same tool 3× in a row
    )
    breaker = CircuitBreaker(config)

    # ── 2. Set up the "client" (swap for real openai.OpenAI() in production)
    client = _mock_openai_client()
    messages = [{"role": "user", "content": "Explain Python asyncio in detail."}]
    observation: str | None = None

    print("LongGuard Raw-Client Example")
    print("=" * 50)

    # ── 3. Agent loop ──────────────────────────────────────────────────────
    for step_number in range(1, 31):
        t0 = time.monotonic()

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )

        latency_ms = (time.monotonic() - t0) * 1000

        # Convert the raw response to an AgentStep in one line ──────────────
        step = AgentStep.from_openai_response(
            response,
            step_number=step_number,
            observation=observation,
            latency_ms=latency_ms,
        )

        print(f"\nStep {step_number} | tokens={step.tokens_used}")
        if step.action:
            print(f"  → Tool: {step.action}({step.action_input})")
        else:
            print(f"  → Response: {step.thought[:80]}...")

        # Check with LongGuard ─────────────────────────────────────────────
        decision = breaker.check(step)

        if decision.action == "kill":
            print(f"\n⛔  KILLED: {decision.reason}")
            break

        if decision.action == "reflect":
            print(f"\n⚠️  LOOP DETECTED — injecting Reflect & Pivot:")
            print(f"   {decision.inject_prompt[:120]}...")
            # In production: append decision.inject_prompt as a system message
            messages.append({"role": "system", "content": decision.inject_prompt})

        if step.action is None:
            # No tool call — the agent gave a final answer
            print("\n✅  Agent completed naturally.")
            break

        # Simulate tool execution (replace with real tool call)
        observation = f"[Tool {step.action} returned: sample result #{step_number}]"
        messages.append({"role": "tool", "content": observation})

    # ── 4. Print the final report ──────────────────────────────────────────
    print("\n" + "=" * 50)
    print(breaker.report.summary())

    # Optionally save the report
    breaker.report.save("run_report.json")
    print("\nReport saved → run_report.json")


if __name__ == "__main__":
    main()
