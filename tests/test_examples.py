"""Automated verification that all repository examples run cleanly."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

ROOT_DIR = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT_DIR / "examples"

EXAMPLE_SCRIPTS = [
    ("custom_detector.py", "LongGuard Custom Detector Example", []),
    ("langchain_agent.py", "LongGuard Manual Step Checking Example", []),
    ("langgraph_basic.py", "LongGuard Basic Example: Simulating a Looping Agent", []),
    ("langgraph_resume.py", "Completed tools:", ["langgraph", "langchain_core"]),
    ("crewai_guarded.py", "recovered", ["crewai"]),
    ("openai_raw_guard.py", "LongGuard Raw-Client Example", []),
]


@pytest.mark.parametrize("script_name, expected_marker, required_modules", EXAMPLE_SCRIPTS)
def test_example_script(
    tmp_path: Path, script_name: str, expected_marker: str, required_modules: list[str]
) -> None:
    """Run example script in an isolated temp directory and verify zero exit code."""
    for module_name in required_modules:
        pytest.importorskip(module_name)

    script_path = EXAMPLES_DIR / script_name
    assert script_path.exists(), f"Example script {script_name} does not exist at {script_path}"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR / "src")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["CREWAI_DISABLE_TELEMETRY"] = "true"
    env["OTEL_SDK_DISABLED"] = "true"
    env["CREWAI_TRACING_ENABLED"] = "false"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=60,
    )

    assert result.returncode == 0, (
        f"Example script {script_name} failed with return code {result.returncode}.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert expected_marker in result.stdout, (
        f"Expected marker '{expected_marker}' not found in stdout of {script_name}.\n"
        f"STDOUT:\n{result.stdout}"
    )
    if script_name == "openai_raw_guard.py":
        assert (tmp_path / "run_report.json").exists(), (
            "openai_raw_guard.py did not produce run_report.json in cwd"
        )
