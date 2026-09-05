# Reflect & Pivot Recovery

When a loop is detected, simply killing the agent is a wasted opportunity. In many cases, the LLM simply needs an explicit signal that its current strategy is failing.

LongGuard’s **Reflect & Pivot** mechanism injects targeted guidance into the agent context to force a course correction.

---

## Default Pivot Templates

LongGuard formats a context-aware prompt depending on which detector triggered:

| Trigger Pattern | Prompt Strategy |
|---|---|
| **`tool_repeat`** | *"You have called `{tool}` multiple times without success. STOP calling this tool. Try a completely different tool, re-scope your goal, or summarize what you have so far."* |
| **`semantic_oscillation`** | *"Your reasoning is cycling between states. Step back, state what you know with certainty, identify the single missing piece of information, and change direction."* |
| **`dead_end_drift`** | *"Your last {steps} actions produced no new information. Provide your best answer with current info or abandon this path entirely."* |
| **`token_velocity`** | *"Your token consumption has spiked dramatically. Summarize concisely in 2-3 sentences and proceed directly to completion."* |

---

## Customizing Pivot Templates

You can override templates globally or per detector:

```python
from longguard import GuardConfig

config = GuardConfig(
    pivot_templates={
        "tool_repeat": (
            "[SYSTEM ALERT] You have repeated {tool} {count} times! "
            "Explain your difficulty to the user instead of trying again."
        )
    }
)
```

### Available Template Variables
- `{tool}`: Name of the repeating tool.
- `{count}`: Number of repetitions.
- `{window}`: Sliding window size.
- `{steps}`: Consecutive dead-end steps.
- `{velocity}` / `{baseline}`: Token consumption metrics.
