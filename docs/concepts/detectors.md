# Loop Detectors

LongGuard includes four battle-tested detectors that analyze different dimensions of an agent's execution.

---

## 1. ToolRepeatDetector

**What it catches:** The agent calls the exact same tool with the exact same arguments repeatedly.

**How it works:**
- Calculates an MD5 fingerprint of `(tool_name, serialized_args)`.
- Tracks fingerprints across a sliding window of recent steps.
- **Smart Polling Awareness:** Legitimate polling (e.g. calling `check_status(id=123)` with changing responses, or calling `search(query="different")`) is **not** flagged.

```python
from longguard import ToolRepeatDetector

detector = ToolRepeatDetector(
    repeat_threshold=3,  # 3 identical calls triggers detection
    window=6,            # Look back over the last 6 steps
)
```

---

## 2. SemanticOscillationDetector

**What it catches:** The agent's chain-of-thought text cycles through the same semantic territory — varying its words slightly, but re-hashing the same thought loop.

**How it works:**
- Computes an embedding for each thought step.
- Calculates variance across recent embeddings in a sliding window. Low variance indicates cognitive deadlock.
- **Embedders:** Uses a fast, deterministic, zero-dependency `HashBasedEmbedder` by default. Optionally upgrade to `SentenceTransformerEmbedder` via `pip install "longguard[embeddings]"`.

```python
from longguard import SemanticOscillationDetector, HashBasedEmbedder

detector = SemanticOscillationDetector(
    window=8,
    variance_threshold=0.15,
    embedder=HashBasedEmbedder(dimension=64),
)
```

---

## 3. DeadEndDriftDetector

**What it catches:** The agent is taking actions, but making zero meaningful progress toward the objective.

**How it works:**
Measures three novelty signals on each step:
1. **Observation Novelty:** Jaccard or cosine similarity of new tool observations vs. prior observations.
2. **Action Diversity:** Switching to an alternative tool indicates exploration.
3. **Thought Novelty:** Introduction of new concepts in reasoning.

If none of these novelty signals appear for `dead_end_threshold` consecutive steps, the detector flags dead-end drift.

```python
from longguard import DeadEndDriftDetector

detector = DeadEndDriftDetector(
    dead_end_threshold=5,
    progress_threshold=0.6,
)
```

---

## 4. TokenVelocityDetector

**What it catches:** Sudden exponential cost spikes in the agent's thoughts or context.

**How it works:**
- Establishes a baseline of tokens-per-step using an exponential moving average (EMA).
- After a short warmup period, if current step velocity exceeds the baseline by `velocity_multiplier` (e.g. 3.0×), it trips detection.

```python
from longguard import TokenVelocityDetector

detector = TokenVelocityDetector(
    velocity_multiplier=3.0,
    window=5,
    warmup=3,
)
```
