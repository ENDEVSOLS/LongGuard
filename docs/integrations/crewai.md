# CrewAI Integration
LongGuard provides first-class support for [CrewAI](https://github.com/crewAIInc/crewAI) multi-agent systems via `CrewGuard` and `add_guard_to_crew`.

When multiple autonomous agents interact in a crew, they are susceptible to:
- **Repetitive tool execution**: Repeatedly querying search or APIs with identical parameters.
- **Delegation loops**: Agent A delegating to Agent B, who delegates back to Agent A.
- **Runaway token costs**: Agents burning through the context budget on dead-end subtasks.

LongGuard monitors CrewAI steps in real time, triggers **Reflect & Pivot** prompt guidance when loops emerge, and gracefully trips the circuit breaker before blowing your budget.

---

## Installation

Install LongGuard with optional CrewAI support:

```bash
pip install "longguard[crewai]"
# or
pip install longguard crewai
```

---

## 1-Line Crew Protection (`add_guard_to_crew`)

The simplest way to guard an entire crew is using `add_guard_to_crew`:

```python
from crewai import Agent, Crew, Task
from longguard import GuardConfig
from longguard.integrations.crewai import add_guard_to_crew

# 1. Define your agents and tasks as normal
researcher = Agent(
    role="Senior Market Analyst",
    goal="Discover emerging trends in generative AI",
    backstory="You are an expert market researcher with deep technical knowledge.",
    tools=[search_tool],
)

task = Task(
    description="Analyze the top 5 AI agent frameworks in 2026.",
    expected_output="A bulleted summary with market share estimates.",
    agent=researcher,
)

crew = Crew(
    agents=[researcher],
    tasks=[task],
)

# 2. Add LongGuard in one line!
crew = add_guard_to_crew(
    crew,
    config=GuardConfig(
        model="gpt-4o",
        max_cost_usd=1.00,             # Hard cap spend at $1.00
        tool_repeat_threshold=3,       # Stop if any agent repeats a tool 3 times
    ),
)

# 3. Kick off execution
result = crew.kickoff()

# 4. Inspect full execution telemetry and cost
guard = crew.__longguard__
print(guard.summary())
```

---

## Agent-Level Protection (`CrewGuard`)

If you want granular control over individual agents in a multi-agent crew:

```python
from crewai import Agent
from longguard import GuardConfig
from longguard.integrations.crewai import CrewGuard

# Create a dedicated guard instance
guard = CrewGuard(GuardConfig(
    tool_repeat_threshold=2,
    max_tokens_per_run=40_000,
))

# Attach step_callback to individual agents
analyst = Agent(
    role="Data Analyst",
    goal="Verify numerical consistency",
    backstory="You double-check calculations and tables.",
    step_callback=guard.step_callback,
)

writer = Agent(
    role="Technical Writer",
    goal="Draft the final executive summary",
    backstory="You craft concise reports.",
    step_callback=guard.step_callback,
)
```

---

## Handling Circuit Breaker Trips

If an agent enters an unrecoverable reasoning loop or exceeds your spend cap, LongGuard raises `GuardTerminatedException`:

```python
from longguard.integrations.crewai import GuardTerminatedException, add_guard_to_crew

crew = add_guard_to_crew(crew, GuardConfig(max_cost_usd=0.50))

try:
    result = crew.kickoff()
except GuardTerminatedException as exc:
    print(f"⛔ Crew execution halted: {exc.reason}")
    print(f"Total spent: ${exc.report.total_cost_usd:.4f}")
    
    # Save the execution audit trail for debugging
    exc.report.save("failed_crew_run.json")
```

---

## Telemetry & Reporting

`CrewGuard` records every thought, tool name, argument hash, and token count:

```python
guard = crew.__longguard__
report = guard.get_report()

print(f"Steps executed: {len(report.steps)}")
print(f"Tokens consumed: {report.total_tokens}")
print(f"Detections triggered: {len(report.detections)}")

# Export audit trail to JSON or YAML
report.save("crew_telemetry.json")
```
