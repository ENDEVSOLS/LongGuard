# Performance & Overhead

LongGuard evaluates agent trajectory steps in-flight. Because breaker evaluations execute synchronously between agent LLM calls and tool executions, minimal overhead is essential.

This page documents the empirical benchmark methodology, measured **p50, p95, and p99** latency per step across all detectors, and instructions for reproducing results locally.

---

## Measured Latency Overview

Overhead is measured on a synthetic 1,000-step execution stream simulating realistic agent interactions (chain-of-thought thoughts, tool invocations, 2–5 KB JSON observations, and occasional 20–50 KB payloads).

The table below reflects baseline measurements using Python 3.12 on Linux (x86_64):

| Detector Configuration | Embedding Backend | p50 (Median) | p95 | p99 | Mean | Unit |
|---|---|---:|---:|---:|---:|:---:|
| **Tool Repeat** | None | 11.0 µs | 20.0 µs | 28.7 µs | 12.8 µs | Microseconds |
| **Semantic Oscillation** | `HashBasedEmbedder` | 182.4 µs | 336.8 µs | 409.5 µs | 205.2 µs | Microseconds |
| **Dead-End Drift** | Jaccard (Lexical) | 92.8 µs | 584.6 µs | 787.0 µs | 147.4 µs | Microseconds |
| **Token Velocity** | None | 14.6 µs | 38.1 µs | 53.4 µs | 17.9 µs | Microseconds |
| **Combined (All 4)** | **`HashBasedEmbedder`** | **352.7 µs** | **846.1 µs** | **1,044.0 µs** | **399.7 µs** | **Microseconds** |

> **Key Takeaway:** With default detectors and zero external ML dependencies (`HashBasedEmbedder`), the combined circuit breaker median check latency is **~350 microseconds (0.35 ms)**. 95% of steps complete in under **850 microseconds (0.85 ms)**.

---

## Embedding Backends: Hash vs Neural Transformers

LongGuard supports two embedding backends for semantic detectors:

1. **Default `HashBasedEmbedder` (Zero-Dependency):**
   * Uses multi-round SHA-256 projection to generate 64-dimensional pseudo-embedding unit vectors.
   * Runs in **sub-millisecond time (~180 µs median)** with zero external dependencies.
   * Ideal for high-throughput production loops, edge environments, and CI pipelines where external neural models add unnecessary latency or memory footprint.

2. **Optional `SentenceTransformerEmbedder` (`longguard[embeddings]`):**
   * Employs PyTorch and Hugging Face sentence-transformers (e.g., `all-MiniLM-L6-v2`) producing dense 384-dimensional embeddings.
   * Requires GPU or multi-core CPU inference; per-step check latency increases to **15–40 milliseconds** on CPU depending on hardware.
   * Provides deeper semantic nuance for open-ended agents where lexical or hash comparisons produce false positives.

---

## Memory Footprint & Scaling

LongGuard maintains bounded memory by design:
* **Sliding Window Buffers:** Detectors store only the most recent $W$ steps (`collections.deque(maxlen=window)`), where $W$ defaults between 5 and 10 steps.
* **Complexity:** Per-step time complexity is bounded by $O(W \cdot D)$ where $D$ is vector dimensionality.
* **No Unbounded Leaks:** Trajectory memory does not grow monotonically across runs; history buffers retain only active sliding windows.

---

## Shared CI Runner Caveats

In continuous integration environments (such as GitHub Actions `ubuntu-latest` runners):
* Virtual machines execute on shared cloud hardware with variable CPU throttling and noisy-neighbor virtualization jitter.
* Percentile timings (specifically **p95 and p99**) measured in CI will typically exhibit higher variance and occasional spikes compared to dedicated bare-metal or production instances.
* Automated CI benchmark jobs run in **report-only mode** to record telemetry artifacts without gating builds on volatile cloud timing.

---

## Reproducing Locally

To run the complete 1,000-step benchmark harness on your local machine:

```bash
# Clone and install development dependencies
git clone https://github.com/ENDEVSOLS/LongGuard.git
cd LongGuard
uv sync --extra dev

# Run 1,000-step benchmark with JSON export
uv run python tests/benchmarks/overhead.py --steps 1000 --output tests/benchmarks/results/local-performance.json
```

To test with the optional neural embedding backend (requires `longguard[embeddings]`):

```bash
uv sync --extra dev --extra embeddings
uv run python tests/benchmarks/overhead.py --backend sentence-transformers --steps 1000
```
