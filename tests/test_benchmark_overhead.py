"""Tests for LongGuard performance overhead benchmark harness."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.run import run_benchmark
from benchmarks.workload import generate_synthetic_stream


def test_synthetic_stream_generation() -> None:
    steps = generate_synthetic_stream(count=30, seed=123)
    assert len(steps) == 30

    for i, step in enumerate(steps, start=1):
        assert step.step_number == i
        assert len(step.thought) >= 30
        assert step.action is not None
        assert step.action_input is not None
        assert step.observation is not None
        assert step.tokens_used > 0
        assert step.latency_ms > 0

    # Test that observation sizes match criteria
    # Step 10 and 20 should be large payloads (> 15 KB)
    assert len(steps[9].observation.encode("utf-8")) > 15 * 1024
    assert len(steps[19].observation.encode("utf-8")) > 15 * 1024
    # Normal step should be between 1.5 KB and 8 KB
    assert 1.5 * 1024 <= len(steps[0].observation.encode("utf-8")) <= 8 * 1024


def test_run_benchmark_execution(tmp_path: Path) -> None:
    output_file = tmp_path / "test_benchmark.json"
    _ = run_benchmark(
        step_count=25,
        warmup_count=5,
        backend="hash",
        seed=99,
        output_path=output_file,
    )

    assert output_file.exists()
    data = json.loads(output_file.read_text(encoding="utf-8"))

    assert "environment" in data
    assert data["environment"]["os"]
    assert data["environment"]["python_version"]
    assert "benchmarks" in data
    assert len(data["benchmarks"]) == 5

    configs = {b["configuration"] for b in data["benchmarks"]}
    assert configs == {
        "tool_repeat",
        "semantic_oscillation",
        "dead_end_drift",
        "token_velocity",
        "combined",
    }

    for b in data["benchmarks"]:
        assert b["backend"] == "hash"
        assert b["steps_measured"] == 25
        assert b["p50_us"] > 0
        assert b["p95_us"] >= b["p50_us"]
        assert b["p99_us"] >= b["p95_us"]
        assert b["mean_us"] > 0
