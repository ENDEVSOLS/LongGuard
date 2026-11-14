"""Execute release gates and write evidence, even if a gate fails."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Build using cached dependencies")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dist_dir = root / "dist"
    if dist_dir.exists():
        for p in dist_dir.iterdir():
            if p.is_file() and p.name != ".gitignore":
                p.unlink()

    junit_path = root / "release-tests.xml"
    commands = {
        "version_check": [sys.executable, "scripts/check_version.py"],
        "links": [sys.executable, "scripts/check_links.py"],
        "ruff": [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "src/",
            "tests/",
            "scripts/",
        ],
        "mypy": [sys.executable, "-m", "mypy", "--python-version", "3.12", "src/longguard"],
        "pytest": [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-q",
            "--cov=longguard",
            "--cov-report=term-missing",
            f"--cov-report=json:{root / 'release-coverage.json'}",
            "--cov-fail-under=95",
            f"--junitxml={junit_path}",
        ],
        "docs": [sys.executable, "-m", "mkdocs", "build", "--strict"],
        "build": ["uv", "build", *(["--offline"] if args.offline else [])],
        "twine": [sys.executable, "-m", "twine", "check", "--strict", "dist/*"],
        "check_dist": [sys.executable, "scripts/check_version.py", "--check-dist"],
    }
    record = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {},
        "checks": {},
        "artifacts_sha256": {},
    }
    # Release verification requires real optional framework dependencies.
    for name in ("langgraph", "langchain-core", "langchain", "crewai"):
        record["dependencies"][name] = importlib.metadata.version(name)
    for name, command in commands.items():
        completed = subprocess.run(command, cwd=root, text=True, capture_output=True)
        record["checks"][name] = {
            "command": command,
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        if name == "pytest" and completed.returncode == 0:
            cases = list(ET.parse(junit_path).iter("testcase"))
            for module in ("test_crewai_execution", "test_langgraph_resume", "test_examples"):
                selected = [case for case in cases if module in case.get("classname", "")]
                if not selected or any(case.find("skipped") is not None for case in selected):
                    record["checks"][name]["exit_code"] = 1
                    record["checks"][name]["stderr"] += f"Required real tests skipped: {module}"
        print(name, record["checks"][name]["exit_code"], flush=True)
        evidence_json = json.dumps(record, indent=2) + "\n"
        (root / "release-evidence.json").write_text(evidence_json, encoding="utf-8")
    coverage_path = root / "release-coverage.json"
    if record["checks"]["pytest"]["exit_code"] == 0 and coverage_path.exists():
        record["coverage"] = json.loads(coverage_path.read_text(encoding="utf-8"))["totals"]
    for path in (root / "dist").glob("*"):
        if path.is_file() and (path.name.endswith(".whl") or path.name.endswith(".tar.gz")):
            record["artifacts_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    evidence_json = json.dumps(record, indent=2) + "\n"
    (root / "release-evidence.json").write_text(evidence_json, encoding="utf-8")
    if any(result["exit_code"] for result in record["checks"].values()):
        raise SystemExit("Release checks failed; inspect release-evidence.json")


if __name__ == "__main__":
    main()
