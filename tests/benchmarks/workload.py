"""Synthetic step-stream workload generator for LongGuard performance benchmarking."""

from __future__ import annotations

import json
import random
from typing import Final

from longguard import AgentStep

TOOL_POOL: Final[list[str]] = [
    "web_search",
    "query_vector_db",
    "fetch_api_endpoint",
    "calculate_statistics",
    "parse_document_tree",
    "summarize_chunk",
    "validate_schema",
    "extract_entities",
    "execute_code_sandbox",
    "render_markdown_table",
]

THOUGHT_TEMPLATES: Final[list[str]] = [
    "Analyzing document section {idx} to extract relevant financial indicators.",
    "Executing database query for entity {idx} and validating schema conformance.",
    "Cross-referencing search results from query {idx} with cached representations.",
    "Transforming raw observation payloads from step {idx} into structured entities.",
    "Formulating next hypothesis regarding parameter {idx} based on feedback.",
    "Evaluating semantic relevance of chunk {idx} relative to prompt constraints.",
    "Filtering noisy telemetry signals from sensor {idx} prior to final aggregation.",
    "Generating intermediate reasoning tokens to address constraint {idx} in space.",
]


def _build_payload(rng: random.Random, min_kb: int, max_kb: int, step_num: int) -> str:
    """Generate a realistic structured JSON observation within the requested KB range."""
    target_bytes = rng.randint(min_kb * 1024, max_kb * 1024)
    records = []
    current_size = 0
    record_id = 1
    while current_size < target_bytes:
        item = {
            "record_id": f"REC-{step_num:04d}-{record_id:04d}",
            "source": f"module-{rng.randint(1, 50)}",
            "score": round(rng.uniform(0.1, 0.99), 4),
            "text": (
                f"Synthetic observation data segment {record_id} for step {step_num}. "
                f"Contextual signals for downstream processing. "
                f"Token hash {rng.getrandbits(32):08x}."
            ),
        }
        chunk = json.dumps(item)
        records.append(item)
        current_size += len(chunk.encode("utf-8"))
        record_id += 1

    return json.dumps({"status": "success", "step": step_num, "data": records}, indent=2)


def generate_synthetic_stream(count: int = 1000, seed: int = 42) -> list[AgentStep]:
    """Generate a reproducible synthetic stream of agent steps for performance benchmarking.

    Characteristics:
    - 1,000 steps (configurable).
    - Short thoughts (50–150 characters) with varied reasoning phrases.
    - 90% realistic standard observations (2–5 KB structured JSON).
    - 10% occasional large observations (20–50 KB structured JSON) every 10th step.
    - Non-looping, realistic tool progression ensuring detectors evaluate normal overhead.

    Args:
        count: Total number of steps to generate.
        seed: Random seed for deterministic reproducibility.

    Returns:
        A list of initialized AgentStep instances.
    """
    rng = random.Random(seed)
    steps: list[AgentStep] = []

    for step_num in range(1, count + 1):
        # Pick alternating tool to avoid triggering ToolRepeatDetector
        tool_name = TOOL_POOL[(step_num - 1) % len(TOOL_POOL)]

        # Pick varied thought
        template = THOUGHT_TEMPLATES[(step_num - 1) % len(THOUGHT_TEMPLATES)]
        thought = template.format(idx=step_num)

        # Action input payload
        action_input = {
            "query": f"search entity query {step_num} {rng.getrandbits(16):04x}",
            "limit": rng.randint(5, 50),
            "filter": f"category_{rng.randint(1, 10)}",
        }

        # Observations: 1 in 10 steps is a large payload (20-50 KB), others are 2-5 KB
        if step_num % 10 == 0:
            observation = _build_payload(rng, min_kb=20, max_kb=50, step_num=step_num)
        else:
            observation = _build_payload(rng, min_kb=2, max_kb=5, step_num=step_num)

        # Realistic tokens and latency
        tokens = rng.randint(300, 1500)
        latency = rng.uniform(80.0, 350.0)

        step = AgentStep(
            step_number=step_num,
            thought=thought,
            action=tool_name,
            action_input=action_input,
            observation=observation,
            tokens_used=tokens,
            latency_ms=latency,
        )
        steps.append(step)

    return steps
