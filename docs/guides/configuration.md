# Configuration Guide

All LongGuard parameters are configured via `GuardConfig`.

---

## Configuration Reference

```python
from longguard import GuardConfig

config = GuardConfig(
    # --- Tool Repeat Detector ---
    tool_repeat_threshold=3,          # N identical calls triggers detection
    tool_repeat_window=6,             # Look-back window size (steps)

    # --- Semantic Oscillation Detector ---
    semantic_variance_threshold=0.15, # Variance threshold below which thoughts are looping
    semantic_window=8,                # Number of thoughts to analyze

    # --- Dead-End Drift Detector ---
    dead_end_threshold=5,             # Consecutive no-progress steps before flagging
    dead_end_progress_threshold=0.6,  # Jaccard/cosine similarity threshold

    # --- Token Velocity Detector ---
    token_velocity_multiplier=3.0,    # Ratio of current velocity to EMA baseline
    token_velocity_window=5,          # Rolling window size
    token_velocity_warmup=3,          # Warmup steps before baseline is active

    # --- Hard Limits ---
    max_tokens_per_run=50_000,        # Hard token spend cap
    max_steps=30,                     # Hard reasoning step limit

    # --- Recovery Settings ---
    max_reflections=2,                # Number of recovery chances before killing
    pivot_templates={},               # Custom prompt templates

    # --- Observability ---
    log_level="WARNING",
    emit_events=True,
)
```

---

## Loading from Files

### From JSON
```python
import json
from longguard import GuardConfig

with open("guard_config.json") as f:
    config = GuardConfig.from_dict(json.load(f))
```

### From YAML
```python
import yaml
from longguard import GuardConfig

with open("guard_config.yaml") as f:
    config = GuardConfig.from_dict(yaml.safe_load(f))
```

---

## Merging Configurations

Easily derive strict or relaxed configs:

```python
base = GuardConfig()
strict = base.merge(max_tokens_per_run=10_000, max_steps=10)
```
