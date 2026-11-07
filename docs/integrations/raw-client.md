# Raw Client Integration (OpenAI & Anthropic)

Use LongGuard directly with the OpenAI or Anthropic Python SDK — no LangGraph or LangChain required.

---

## How It Works

LongGuard provides two class methods on `AgentStep` that extract all necessary
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
    model="claude-sonnet-5-5",
    max_cost_usd=0.50,
    max_steps=30,
))

messages = [{"role": "user", "content": "Analyze this dataset and summarize key findings."}]
observation = None

for i in range(1, 31):
    response = client.messages.create(
        model="claude-sonnet-5-5",
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

LongGuard's built-in pricing table was verified against official provider sources on
2026-10-09. It uses standard, uncached text-token prices at the base context tier.
See the [pricing sources and availability audit](../guides/model-pricing-sources.md)
for supported providers, removals, retirement dates, and promotional price expiry dates.

The selected registry excludes GPT-4/Turbo, GPT-3.5 Turbo, the o-series, Gemini 2.5,
Command R+, and the March 2024 Command R releases. Use `list_supported_models()`
to check whether a model has a built-in price.

Long-context premiums, caching, batch/priority rates, multimodal usage, and tool fees
require user-supplied prices or separate accounting. Removed model IDs return `None`;
provide explicit rates when estimating historical usage or enterprise contracts.

```python
from longguard import list_supported_models, compute_cost

# List all models with known pricing
print(list_supported_models())
# Includes 'claude-sonnet-5-5', 'gemini-3.8-flash', 'gpt-6.1-sol', ...

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


## Stop with pending tool calls

Converters preserve their existing first-action summary. Cleanup must inspect the original
provider conversation to preserve every tool call identifier:

```python
from longguard.integrations.messages import resolve_pending_tool_calls

# First stop dispatching new tools. Append the actual assistant response to messages.
cleanup = resolve_pending_tool_calls(messages, decision.reason, protocol="openai")
messages = cleanup.messages
# Append cleanup.dispositions to your report if desired.
```

Use `protocol="anthropic"` for Anthropic tool-use blocks. Completed results remain unchanged;
unstarted calls receive failed/skipped results and never a fabricated success. If calls are
already running, pass `running_ids={...}`: they remain unresolved until real completion.
Do not continue the provider conversation with unresolved running calls. The helper itself
never cancels or executes tools, and assumes a valid conversation prefix.
