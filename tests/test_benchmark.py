"""The benchmark CLI measures actual detector outcomes on labeled traces."""

import json
import subprocess
import sys
from pathlib import Path


def test_cli_reports_counts_and_undefined_denominators(tmp_path):
    corpus = {
        "revision": "test",
        "assessment_unit": "trace",
        "labels_review": {
            "status": "test_fixture",
        },
        "traces": [
            {
                "id": "loop",
                "family": "loop",
                "split": "evaluation",
                "detector": "tool_repeat",
                "positive": True,
                "timely_by": 3,
                "steps": [
                    {"step_number": n, "thought": "repeat", "action": "search", "tokens_used": 1}
                    for n in (1, 2, 3)
                ],
            },
            {
                "id": "healthy",
                "family": "healthy",
                "split": "evaluation",
                "detector": "tool_repeat",
                "positive": False,
                "timely_by": 1,
                "steps": [
                    {"step_number": 1, "thought": "finish", "tokens_used": 1},
                ],
            },
        ],
    }
    fixture = tmp_path / "corpus.json"
    output = tmp_path / "results.json"
    fixture.write_text(json.dumps(corpus))
    subprocess.run(
        [
            sys.executable,
            "tests/benchmarks/run.py",
            "--corpus",
            str(fixture),
            "--output",
            str(output),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    results = json.loads(output.read_text(encoding="utf-8"))
    assert results["detectors"]["tool_repeat"]["counts"] == {"tp": 1, "fp": 0, "tn": 1, "fn": 0}
    assert results["detectors"]["tool_repeat"]["precision"] == 1.0
    assert results["detectors"]["semantic_oscillation"]["precision"] is None
    assert len(results["traces"]) == 2


def test_cli_uses_bundled_corpus_by_default(tmp_path):
    script = Path(__file__).resolve().parent / "benchmarks" / "run.py"
    output = tmp_path / "results.json"
    subprocess.run(
        [sys.executable, str(script), "--output", str(output)],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    results = json.loads(output.read_text(encoding="utf-8"))
    assert len(results["traces"]) == 32
    assert all(not result["errors"] for result in results["detectors"].values())
