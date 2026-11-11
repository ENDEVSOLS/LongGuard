"""Generate literal release notes from a verified tag, package, and check record."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tomllib
from pathlib import Path

from packaging.version import Version


def generate(root: Path, tag: str, evidence: dict) -> tuple[str, bool]:
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+(?:rc[0-9]+|\.dev[0-9]+)?", tag):
        raise ValueError("Use a version tag such as v0.2.0rc1")
    version = Version(tag[1:])
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    if str(version) != metadata["project"]["version"]:
        raise ValueError("Release tag does not match package version")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if evidence.get("commit") != commit:
        raise ValueError("Check evidence does not identify the checked-out source commit")
    required = {"ruff", "mypy", "pytest", "docs", "build"}
    checks = evidence.get("checks", {})
    if not required.issubset(checks) or any(checks[k]["exit_code"] != 0 for k in required):
        raise ValueError("Release checks are missing or failed")
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    match = re.search(
        rf"^## \[{re.escape(str(version))}\][^\n]*\n(.*?)(?=^## |\Z)",
        changelog,
        re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise ValueError("An explicit changelog entry for the release is required")
    candidate = version.is_prerelease or version.is_devrelease
    flag = " --pre" if candidate else ""
    body = (
        f"# LongGuard {tag}\n\n{match.group(1).strip()}\n\n"
        f"## Install the exact version\n\n```bash\npip install{flag} longguard=={version}\n```\n\n"
        f"Source commit: `{commit}`. Executed checks and commands are attached in "
        "`release-evidence.json`; framework and platform coverage is limited to that record.\n\n"
        "Detector benchmark labels and release acceptance require maintainer review. "
        "No production adoption, provider savings, or untested platform claims are implied.\n"
    )
    return body, candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    body, candidate = generate(root, args.tag, evidence)
    args.output.write_text(body, encoding="utf-8")
    if args.github_output:
        with args.github_output.open("a") as stream:
            stream.write(f"tag={args.tag}\nprerelease={str(candidate).lower()}\n")
            stream.write(f"latest={str(not candidate).lower()}\n")


if __name__ == "__main__":
    main()
