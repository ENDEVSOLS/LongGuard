# Universal `@guarded` Decorator

The `@guarded` decorator brings LongGuard's circuit breaker protection to **any Python function, LLM SDK call, generator, or async workflow** with zero framework coupling.

---

## Why `@guarded`?

Not every agent uses LangGraph or LangChain. If you build custom `while` loops, proprietary micro-agents, or standalone tool execution functions, `@guarded` wraps your logic in a single line:

- 🛡️ **Prevents Runaway Loops**: Automatically tracks thoughts, actions, and observations across repeated function calls.
- 💵 **Enforces Spend Limits**: Automatically computes dollar costs and trips on `max_cost_usd`.
- ⚡ **Zero Framework Dependencies**: Works with pure Python, `openai`, `anthropic`, custom generators, or async coroutines.
- 📊 **Step Telemetry**: Preserves a full `GuardReport` accessible directly via `fn.report`.

---

## 1. Quick Example: Tool / Agent Step Function

Decorate your step execution function with detection parameters:

```python
from longguard import guarded, CircuitBreakerTrippedError

@guarded(max_cost_usd=0.50, tool_repeat_threshold=3)
def execute_agent_step(thought: str, action: str, action_input: dict, observation: str):
    # Your business logic / tool invocation here
    return run_tool(action, action_input)

# Execute steps in your agent loop
try:
    for thought, tool, args in agent_plan:
        result = execute_agent_step(
            thought=thought,
            action=tool,
            action_input=args,
            observation=last_result,
        )
except CircuitBreakerTrippedError as exc:
    print(f"Agent halted safely: {exc.reason}")
    print(exc.report.summary())
```

---

## 2. Decorating OpenAI / Anthropic SDK Calls

Pass LLM responses directly—LongGuard automatically parses response objects and token counts:

```python
import openai
from longguard import guarded

client = openai.OpenAI()

@guarded(model="gpt-4o", max_cost_usd=1.00)
def call_model(messages: list[dict]):
    return client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )

# When called, token usage and dollar cost are automatically accumulated
response = call_model([{"role": "user", "content": "Analyze market data"}])

# Inspect telemetry
print(call_model.report.summary())
```

---

## 3. Graceful Fallbacks (`on_kill="return"`)

If you prefer not to catch exceptions, set `on_kill="return"` or pass a custom callable:

```python
@guarded(tool_repeat_threshold=3, on_kill="return")
def search_database(query: str):
    return db.query(query)

result = search_database("SELECT * FROM users")
# If tripped, returns a safe fallback message instead of raising:
# "Agent terminated by LongGuard: Tool repeat detected: search_database called 3 times"
```

You can also provide a custom callable for custom fallback payloads:

```python
def handle_trip(decision, *args, **kwargs):
    return {"error": True, "reason": decision.reason, "status": "circuit_open"}

@guarded(on_kill=handle_trip)
def run_action(*args, **kwargs):
    ...
```

---

## 4. Generator Support (Streaming & Step Loops)

`@guarded` seamlessly protects Python generators:

```python
from longguard import AgentStep, guarded

@guarded(max_steps=10)
def stream_agent_steps():
    for i in range(100):
        yield AgentStep(
            step_number=i + 1,
            thought=f"Reasoning step {i}",
            action="search" if i % 2 == 0 else "analyze",
            action_input={"query": "financials"},
        )

# Iteration stops automatically when the circuit breaker trips
for step in stream_agent_steps():
    print(f"Step {step.step_number} completed")
```

---

## 5. Async Functions

`@guarded` supports native Python `async def` functions:

```python
@guarded(max_cost_usd=0.75, tool_repeat_threshold=3)
async def async_agent_step(thought: str, action: str, action_input: dict):
    result = await async_tool_runner(action, action_input)
    return result
```

---

## Telemetry & State Management

Every decorated function exposes its state:

```python
# Access current report
report = my_function.report

# Access underlying CircuitBreaker instance
breaker = my_function.breaker

# Reset telemetry and state for a fresh execution
my_function.reset()

# Check if a reflection prompt was triggered
if my_function.last_reflection_prompt:
    print("Injected pivot:", my_function.last_reflection_prompt)
```
