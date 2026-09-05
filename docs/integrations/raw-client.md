# Raw Client Integration (OpenAI & Anthropic)

Use LongGuard directly with the OpenAI or Anthropic Python SDK — no LangGraph or LangChain required.

---

## How It Works

LongGuard v0.1.3 adds two class methods to `AgentStep` that extract all necessary
information from a raw SDK response using duck-typing (no hard SDK dependency):

```python
AgentStep.from_openai_response(response, step_number, *, observation=None, latency_ms=0.0)
AgentStep.from_anthropic_response(response, step_number, *, observation=None, latency_ms=0.0)
```

Both methods extract:

| Field | OpenAI | Anthropic |
|---|---|---|
| `thought` | `choices[0].message.content` | All `text` content blocks joined |
| `action` | `tool_calls[0].function.name` | First `tool_use` block name |
| `action_input` | Parsed JSON from `tool_calls[0].function.arguments` | `tool_use` block input dict |
| `tokens_used` | `usage.total_tokens` | `usage.input_tokens + output_tokens` |

---

## OpenAI Example

```python
import openai
from longguard import AgentStep, CircuitBreaker, GuardConfig

client = openai.OpenAI()
breaker = CircuitBreaker(GuardConfig(
    model="gpt-4o",       # enables dollar-cost tracking
    max_cost_usd=0.50,    # hard-kill if run exceeds $0.50
    max_steps=30,
))

messages = [{"role": "user", "content": "Research and summarize AI trends in 2025."}]
observation = None

for i in range(1, 31):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=[...],  # your tool definitions
    )

    # One-line conversion
    step = AgentStep.from_openai_response(
        response,
        step_number=i,
        observation=observation,
    )

    decision = breaker.check(step)

    if decision.action == "kill":
        print(f"⛔ Halted: {decision.reason}")
        break
    elif decision.action == "reflect":
        messages.append({"role": "system", "content": decision.inject_prompt})

    if step.action is None:
        break  # agent gave final answer

    # Run the tool and get the observation for next step
    observation = your_tool_runner(step.action, step.action_input)
    messages.append({"role": "tool", "content": observation})

print(breaker.report.summary())
# → Estimated Cost: $0.0143 USD (gpt-4o)
```

---

## Anthropic Example

```python
import anthropic
from longguard import AgentStep, CircuitBreaker, GuardConfig

client = anthropic.Anthropic()
breaker = CircuitBreaker(GuardConfig(
    model="claude-3-5-sonnet",
    max_cost_usd=0.50,
    max_steps=30,
))

messages = [{"role": "user", "content": "Analyze this dataset and summarize key findings."}]
observation = None

for i in range(1, 31):
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2048,
        tools=[...],
        messages=messages,
    )

    step = AgentStep.from_anthropic_response(
        response,
        step_number=i,
        observation=observation,
    )

    decision = breaker.check(step)

    if decision.action == "kill":
        print(f"⛔ Halted: {decision.reason}")
        break
    elif decision.action == "reflect":
        messages.append({"role": "user", "content": decision.inject_prompt})

    if step.action is None:
        break

    observation = your_tool_runner(step.action, step.action_input)
    messages.append({"role": "user", "content": [{"type": "tool_result", "content": observation}]})

print(breaker.report.summary())
```

---

## Supported Models & Pricing

LongGuard ships with a built-in pricing table for 40+ models updated as of v0.1.3:

```python
from longguard import list_supported_models, compute_cost

# List all models with known pricing
print(list_supported_models())
# ['claude-3-5-haiku', 'claude-3-5-sonnet', 'gemini-1.5-flash', 'gpt-4o', ...]

# Compute cost for a specific workload
cost = compute_cost("gpt-4o", input_tokens=10_000, output_tokens=3_000)
# $0.055
```

### Pricing Resolution Order

1. **User-supplied** — `cost_per_input_token` + `cost_per_output_token` in `GuardConfig`
2. **Built-in table** — automatic lookup by `model` name (case-insensitive)
3. **Graceful fallback** — `estimated_cost_usd = None` if model not recognised; `max_cost_usd` cap silently skipped

For models not yet in the table, supply your own prices:

```python
config = GuardConfig(
    cost_per_input_token=3e-6,   # look up from provider pricing page
    cost_per_output_token=15e-6,
    max_cost_usd=1.00,
)
```

---

## Runnable Example

A complete self-contained demo (runs without an API key via mock):

```bash
python examples/openai_raw_guard.py
```
