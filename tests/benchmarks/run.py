"""Local detector/circuit evaluation; no provider calls, uploads, or mocked scores."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from longguard import (
    AgentStep,
    CircuitBreaker,
    DeadEndDriftDetector,
    GuardConfig,
    HashBasedEmbedder,
    SemanticOscillationDetector,
    SentenceTransformerEmbedder,
    TokenVelocityDetector,
    ToolRepeatDetector,
)

NAMES = ("tool_repeat", "semantic_oscillation", "dead_end_drift", "token_velocity")


def source_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    if args.backend == "sentence-transformers" and not (args.model_path and args.model_revision):
        raise ValueError("Provide a local model snapshot and its pinned model revision")
    start = time.perf_counter_ns()
    embedder = (
        SentenceTransformerEmbedder(str(args.model_path))
        if args.backend == "sentence-transformers"
        else HashBasedEmbedder()
    )
    cold_ms = (time.perf_counter_ns() - start) / 1e6
    config = GuardConfig()

    def detectors() -> dict[str, Any]:
        return {
            "tool_repeat": ToolRepeatDetector(
                config.tool_repeat_threshold, config.tool_repeat_window
            ),
            "semantic_oscillation": SemanticOscillationDetector(
                config.semantic_window,
                config.semantic_variance_threshold,
                embedder,
            ),
            "dead_end_drift": DeadEndDriftDetector(
                config.dead_end_threshold,
                config.dead_end_progress_threshold,
                embedder if args.backend == "sentence-transformers" else None,
            ),
            "token_velocity": TokenVelocityDetector(
                config.token_velocity_multiplier,
                config.token_velocity_window,
                config.token_velocity_warmup,
            ),
        }

    class EvaluationBreaker(CircuitBreaker):
        """Reuse actual policy with the explicitly selected real embedding backend."""

        @staticmethod
        def _create_shared_embedder() -> Any:
            return embedder if args.backend == "sentence-transformers" else None

    families: dict[str, str] = {}
    ids: set[str] = set()
    for trace in corpus["traces"]:
        if trace["id"] in ids:
            raise ValueError("Duplicate trace identifier")
        ids.add(trace["id"])
        family = trace["family"]
        split = trace["split"]
        if family in families and families[family] != split:
            raise ValueError("Related trace families cross the calibration/evaluation boundary")
        families[family] = split

    summaries = {
        name: {"counts": {"tp": 0, "fp": 0, "tn": 0, "fn": 0}, "errors": [], "skipped": 0}
        for name in NAMES
    }
    details: list[dict[str, Any]] = []
    latency: list[float] = []
    initialization: list[float] = []
    embedding_latency: list[float] = []
    for trace in corpus["traces"]:
        if trace["split"] != "evaluation":
            continue
        name = trace["detector"]
        detector = detectors()[name]
        first: int | None = None
        try:
            for values in trace["steps"]:
                step = AgentStep.from_dict(values)
                if detector.analyze(step).detected and first is None:
                    first = step.step_number
            detected = first is not None and first <= trace["timely_by"]
            key = (
                ("tp" if detected else "fn")
                if trace["positive"]
                else ("fp" if first is not None else "tn")
            )
            start = time.perf_counter_ns()
            breaker = EvaluationBreaker(config)
            initialization.append((time.perf_counter_ns() - start) / 1e6)
            actions: list[str] = []
            for values in trace["steps"]:
                step = AgentStep.from_dict(values)
                start = time.perf_counter_ns()
                embedder.embed(step.thought or "")
                embedding_latency.append((time.perf_counter_ns() - start) / 1e6)
                start = time.perf_counter_ns()
                decision = breaker.check(step)
                latency.append((time.perf_counter_ns() - start) / 1e6)
                actions.append(decision.action)
                if decision.action == "kill":
                    break
            summaries[name]["counts"][key] += 1
            details.append(
                {
                    "id": trace["id"],
                    "detector": name,
                    "positive": trace["positive"],
                    "classification": key,
                    "first_detection_step": first,
                    "timely_by": trace["timely_by"],
                    "circuit_actions": actions,
                    "circuit_reason": breaker.report.kill_reason,
                    "recovery": "clean_after_reflection"
                    if "reflect" in actions and actions[-1] == "continue"
                    else None,
                    "circuit_report": breaker.report.to_dict(),
                }
            )
        except Exception as exc:
            summaries[name]["errors"].append({"id": trace["id"], "error": str(exc)})

    for summary in summaries.values():
        counts = summary["counts"]
        tp, fp, tn, fn = (counts[k] for k in ("tp", "fp", "tn", "fn"))
        summary.update(
            {
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "false_positive_rate": fp / (fp + tn) if fp + tn else None,
                "denominators": {"flagged": tp + fp, "positive": tp + fn, "negative": fp + tn},
                "evaluated": tp + fp + tn + fn,
            }
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    import longguard

    package_root = Path(longguard.__file__).parent
    packages = {}
    for name in ("longguard", "numpy", "sentence-transformers", "torch", "langgraph", "crewai"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "commit": commit,
        "working_tree_dirty": dirty,
        "source_sha256": source_digest(package_root),
        "dataset_revision": corpus["revision"],
        "dataset_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "assessment_unit": corpus["assessment_unit"],
        "timely_detection_rule": corpus.get("timely_detection_rule", "per-trace timely_by"),
        "labels_review": corpus["labels_review"],
        "thresholds": config.to_dict(),
        "backend": args.backend,
        "model_revision": args.model_revision,
        "model_path": str(args.model_path) if args.model_path else None,
        "seed": None,
        "dependencies": packages,
        "source_version": longguard.__version__,
        "source_path": str(package_root),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "command": args.command,
        "detectors": summaries,
        "traces": details,
        "latency": {
            "workload": "actual guard.check over held-out fixture steps; excludes models/tools",
            "embedding_initialization_ms": cold_ms,
            "guard_initialization_mean_ms": statistics.mean(initialization)
            if initialization
            else None,
            "standalone_embedding_mean_ms": statistics.mean(embedding_latency)
            if embedding_latency
            else None,
            "standalone_embedding_workload": "same observable thought texts; separate from checks",
            "warm_checks": len(latency),
            "warm_mean_ms": statistics.mean(latency) if latency else None,
            "warm_median_ms": statistics.median(latency) if latency else None,
            "warm_max_ms": max(latency) if latency else None,
            "embedding_cost": "included in warm guard checks; backend is identified separately",
        },
        "limitations": [
            "Small correlated synthetic corpus; counts are per trace, "
            "not independent production incidents.",
            "Human label review is pending unless the corpus review record says otherwise.",
            "Default thresholds are unchanged; calibration traces are excluded "
            "from held-out metrics.",
            "Circuit outcomes use all detectors, independently of the trace's single target label.",
            "Raw detector metrics do not prove actual provider/tool savings "
            "or production adoption.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path(__file__).with_name("corpus.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("hash", "sentence-transformers"), default="hash")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-revision")
    args = parser.parse_args()
    import sys

    args.command = " ".join([sys.executable, *sys.argv])
    results = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    for name, result in results["detectors"].items():
        print(
            name, result["counts"], "precision=", result["precision"], "recall=", result["recall"]
        )
    if any(result["errors"] for result in results["detectors"].values()):
        raise SystemExit("Benchmark contains errors; inspect saved results")


if __name__ == "__main__":
    main()
