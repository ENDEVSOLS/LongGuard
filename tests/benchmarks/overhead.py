"""Benchmark CLI harness for LongGuard per-step overhead evaluation.

Measures breaker.check() latency (p50, p95, p99) across detectors individually
and combined using synthetic step streams.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

import longguard
from longguard import (
    AgentStep,
    CircuitBreaker,
    DeadEndDriftDetector,
    GuardConfig,
    HashBasedEmbedder,
    SemanticOscillationDetector,
    TokenVelocityDetector,
    ToolRepeatDetector,
)
from longguard.core.breaker import BreakerState

try:
    from .workload import generate_synthetic_stream
except ImportError:
    try:
        from workload import generate_synthetic_stream  # type: ignore[import-not-found]
    except ImportError:
        from tests.benchmarks.workload import (
            generate_synthetic_stream,  # type: ignore[import-not-found]
        )

CONFIG_NAMES = (
    "tool_repeat",
    "semantic_oscillation",
    "dead_end_drift",
    "token_velocity",
    "combined",
)


def create_breaker_for_config(
    name: str,
    backend: str = "hash",
    model_path: str | None = None,
) -> CircuitBreaker:
    """Instantiate a CircuitBreaker configured with relaxed caps and specific detector targets.

    Hard caps are relaxed (max_steps=100,000, max_tokens=100M) and semantic variance threshold
    is aligned with hash dimensioning so the harness measures full detector evaluation latency
    across the complete 1,000-step workload without premature termination.
    """
    config = GuardConfig(
        max_steps=100_000,
        max_tokens_per_run=100_000_000,
        max_cost_usd=None,
        semantic_variance_threshold=0.005 if backend == "hash" else 0.15,
    )
    breaker = CircuitBreaker(config=config)

    # Embedder selection
    embedder: Any
    if backend == "sentence-transformers":
        from longguard import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(model_name_or_path=model_path or "all-MiniLM-L6-v2")
    else:
        embedder = HashBasedEmbedder()

    # Configure specific detectors
    if name == "tool_repeat":
        breaker.detectors = [
            ToolRepeatDetector(
                repeat_threshold=config.tool_repeat_threshold,
                window=config.tool_repeat_window,
            )
        ]
    elif name == "semantic_oscillation":
        breaker.detectors = [
            SemanticOscillationDetector(
                window=config.semantic_window,
                variance_threshold=config.semantic_variance_threshold,
                embedder=embedder,
            )
        ]
    elif name == "dead_end_drift":
        breaker.detectors = [
            DeadEndDriftDetector(
                dead_end_threshold=config.dead_end_threshold,
                progress_threshold=config.dead_end_progress_threshold,
                embedder=embedder if backend == "sentence-transformers" else None,
            )
        ]
    elif name == "token_velocity":
        breaker.detectors = [
            TokenVelocityDetector(
                velocity_multiplier=config.token_velocity_multiplier,
                window=config.token_velocity_window,
                warmup=config.token_velocity_warmup,
            )
        ]
    elif name == "combined":
        breaker.detectors = [
            ToolRepeatDetector(
                repeat_threshold=config.tool_repeat_threshold,
                window=config.tool_repeat_window,
            ),
            SemanticOscillationDetector(
                window=config.semantic_window,
                variance_threshold=config.semantic_variance_threshold,
                embedder=embedder,
            ),
            DeadEndDriftDetector(
                dead_end_threshold=config.dead_end_threshold,
                progress_threshold=config.dead_end_progress_threshold,
                embedder=embedder if backend == "sentence-transformers" else None,
            ),
            TokenVelocityDetector(
                velocity_multiplier=config.token_velocity_multiplier,
                window=config.token_velocity_window,
                warmup=config.token_velocity_warmup,
            ),
        ]
    else:
        raise ValueError(f"Unknown benchmark configuration: {name}")

    return breaker


def measure_configuration(
    name: str,
    steps: list[AgentStep],
    backend: str = "hash",
    warmup_steps: int = 50,
    model_path: str | None = None,
) -> dict[str, Any]:
    """Measure per-step check latency for a given configuration."""
    breaker = create_breaker_for_config(name=name, backend=backend, model_path=model_path)

    # Warmup phase (not timed)
    if warmup_steps > 0 and len(steps) >= warmup_steps:
        for w_step in steps[:warmup_steps]:
            breaker.check(w_step)
            breaker.state = BreakerState.CLOSED

    # Fresh breaker for timed execution
    breaker = create_breaker_for_config(name=name, backend=backend, model_path=model_path)
    latencies_us: list[float] = []

    for step in steps:
        # Guarantee breaker remains active so every step executes the full detector loop
        breaker.state = BreakerState.CLOSED
        breaker._reflection_count = 0

        t0 = time.perf_counter_ns()
        _ = breaker.check(step)
        t1 = time.perf_counter_ns()

        latencies_us.append((t1 - t0) / 1000.0)  # Microseconds

    arr = np.array(latencies_us)
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])

    return {
        "configuration": name,
        "backend": backend,
        "steps_measured": len(latencies_us),
        "p50_us": float(round(p50, 2)),
        "p95_us": float(round(p95, 2)),
        "p99_us": float(round(p99, 2)),
        "mean_us": float(round(float(np.mean(arr)), 2)),
        "stddev_us": float(round(float(np.std(arr)), 2)),
        "min_us": float(round(float(np.min(arr)), 2)),
        "max_us": float(round(float(np.max(arr)), 2)),
        "p50_ms": float(round(p50 / 1000.0, 4)),
        "p95_ms": float(round(p95 / 1000.0, 4)),
        "p99_ms": float(round(p99 / 1000.0, 4)),
    }


def run_benchmark(
    step_count: int = 1000,
    warmup_count: int = 50,
    backend: str = "all",
    seed: int = 42,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run full benchmark across requested configurations and export report."""
    start_time = time.perf_counter()
    timestamp = datetime.datetime.now(datetime.UTC).isoformat()

    print(f"Generating synthetic workload ({step_count} steps, seed={seed})...")
    steps = generate_synthetic_stream(count=step_count, seed=seed)

    backends_to_test = []
    if backend in ("all", "hash"):
        backends_to_test.append("hash")
    if backend in ("all", "sentence-transformers"):
        try:
            from longguard import SentenceTransformerEmbedder

            SentenceTransformerEmbedder()
            backends_to_test.append("sentence-transformers")
        except Exception as exc:
            print(f"Note: Skipping sentence-transformers backend ({exc})")

    results: list[dict[str, Any]] = []

    print("\nRunning overhead benchmark...")
    print("-" * 75)
    header = (
        f"{'Configuration':<24} {'Backend':<12} "
        f"{'p50 (µs)':<10} {'p95 (µs)':<10} {'p99 (µs)':<10} {'Mean (µs)':<10}"
    )
    print(header)
    print("-" * 75)

    for b in backends_to_test:
        for cfg in CONFIG_NAMES:
            res = measure_configuration(
                name=cfg,
                steps=steps,
                backend=b,
                warmup_steps=warmup_count,
            )
            results.append(res)
            row = (
                f"{res['configuration']:<24} {res['backend']:<12} "
                f"{res['p50_us']:<10.1f} {res['p95_us']:<10.1f} "
                f"{res['p99_us']:<10.1f} {res['mean_us']:<10.1f}"
            )
            print(row)

    total_duration = time.perf_counter() - start_time
    print("-" * 75)
    print(f"Benchmark completed in {total_duration:.2f} seconds.")

    environment = {
        "os": platform.system(),
        "os_release": platform.release(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count() or 1,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "longguard_version": getattr(longguard, "__version__", "unknown"),
        "timestamp_utc": timestamp,
        "total_duration_seconds": round(total_duration, 3),
    }

    report = {
        "version": getattr(longguard, "__version__", "unknown"),
        "timestamp": timestamp,
        "environment": environment,
        "workload": {
            "steps": step_count,
            "warmup_steps": warmup_count,
            "seed": seed,
        },
        "benchmarks": results,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Results saved to: {output_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="LongGuard Overhead Benchmark Harness")
    parser.add_argument(
        "--steps", type=int, default=1000, help="Number of synthetic steps (default: 1000)"
    )
    parser.add_argument(
        "--warmup", type=int, default=50, help="Number of warmup steps (default: 50)"
    )
    parser.add_argument(
        "--backend",
        choices=["all", "hash", "sentence-transformers"],
        default="all",
        help="Embedding backend to benchmark (default: all)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Workload generation seed (default: 42)"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Path to save JSON benchmark report"
    )

    args = parser.parse_args()
    run_benchmark(
        step_count=args.steps,
        warmup_count=args.warmup,
        backend=args.backend,
        seed=args.seed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
