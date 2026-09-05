# GuardReport & Telemetry

Every agent execution produces a structured `GuardReport` snapshot.

---

## Accessing the Report

```python
# In LangGraph:
guard = workflow.__longguard__
report = guard.get_report()

# Or directly on CircuitBreaker:
report = breaker.report
```

---

## Output Formats

### 1. Human-Readable Summary
```python
print(report.summary())
```
Output:
```
=== LongGuard Run Report ===
Total Steps: 5
Total Tokens: 1,842
Final State: open
Detections: 2
Reflections Injected: 1
Kill Reason: reflection_failed: tool_repeat (confidence: 100%)

Detection Details:
  Step 4: tool_repeat (confidence: 100%)
  Step 5: tool_repeat (confidence: 100%)
```

### 2. JSON Serialization (Logging & Metrics)
```python
json_str = report.to_json()
```

### 3. Python Dictionary
```python
data = report.to_dict()
# {
#   "total_steps": 5,
#   "total_tokens": 1842,
#   "final_state": "open",
#   "kill_reason": "reflection_failed: tool_repeat...",
#   "detections": [...],
#   "step_timeline": [...]
# }
```

---

## Event Callbacks

Stream events to LangSmith, Datadog, or custom webhooks:

```python
def on_guard_event(event_name: str, payload: dict):
    print(f"LongGuard Event [{event_name}]: {payload}")

breaker = CircuitBreaker(
    config=GuardConfig(emit_events=True),
    event_callback=on_guard_event,
)

# Emits:
# - "detection": loop pattern detected
# - "reflect": pivot prompt injected
# - "kill": agent killed
# - "recovered": circuit recovered to CLOSED
```
