# LangChain Integration

LongGuard integrates with LangChain via the `GuardedAgentExecutor` middleware.

---

## AgentExecutor Middleware

Wrap any standard LangChain `AgentExecutor`:

```python
from langchain.agents import AgentExecutor
from longguard.integrations.langchain import GuardedAgentExecutor, GuardTerminatedException
from longguard import GuardConfig

# 1. Create your standard executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
)

# 2. Wrap with LongGuard
guarded = GuardedAgentExecutor(agent_executor, GuardConfig(
    max_tokens_per_run=40_000,
    max_steps=20,
))

# 3. Run safely
try:
    result = guarded.run("Summarize the quarterly financial report")
    print(result)
except GuardTerminatedException as exc:
    print(f"Agent terminated by LongGuard: {exc.reason}")
    print(exc.report.summary())
```

---

## Manual Step Checking

If you are implementing a custom agent loop rather than `AgentExecutor`:

```python
decision = guarded.check_step(action=agent_action, observation=tool_output)

if decision.action == "kill":
    raise GuardTerminatedException(decision.reason, decision.report)
```
