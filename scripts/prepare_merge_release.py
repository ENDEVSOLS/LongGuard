"""Prepare a versioned release for an already merged pull request.

Supports two modes controlled by the RELEASE_STABLE environment variable:
- Dev release (default):  X.Y.Z.devN  where N = PR number.
- Stable release:         X.Y.Z       strips the .devN suffix entirely.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tomllib
from pathlib import Path


def prepare(root: Path, pr: dict, *, stable: bool = False) -> str:
    if pr.get("merged") is not True or pr["base"]["ref"] != "main":
        raise ValueError("Only merged pull requests targeting main can release")
    number = pr["number"]
    if type(number) is not int or number <= 0:
        raise ValueError("A positive pull request number is required")

    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    source = pr["merge_commit_sha"]
    if git("rev-parse", "HEAD") != source:
        raise ValueError("Checkout must match the merged pull request commit")
    metadata = tomllib.loads((root / "pyproject.toml").read_text())
    current = metadata["project"]["version"]
    match = re.fullmatch(r"(\d+\.\d+\.\d+)\.dev\d+", current)
    if match is None:
        raise ValueError("Automatic releases require an X.Y.Z.devN source version")
    base_version = match[1]

    if stable:
        version = base_version
    else:
        version = f"{base_version}.dev{number}"
    tag = f"v{version}"

    existing = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/tags/{tag}"], cwd=root
    )
    if existing.returncode == 0:
        if git("rev-parse", f"{tag}^{{commit}}^") != source:
            raise ValueError("Existing release tag belongs to a different source commit")
        return tag
    if existing.returncode != 1:
        raise ValueError("Unable to inspect release tags")

    replacements = {
        "pyproject.toml": (f'version = "{current}"', f'version = "{version}"'),
        "src/longguard/__init__.py": (f'__version__ = "{current}"', f'__version__ = "{version}"'),
        "uv.lock": (
            f'name = "longguard"\nversion = "{current}"',
            f'name = "longguard"\nversion = "{version}"',
        ),
    }
    updates = {}
    for name, (old, new) in replacements.items():
        content = (root / name).read_text()
        if content.count(old) != 1:
            raise ValueError(f"Expected exactly one matching version in {name}")
        updates[name] = content.replace(old, new, 1)

    if stable:
        # For stable releases, the CHANGELOG already has the entry under [X.Y.Z].
        # No new entry is added; the existing entry is used as-is.
        changelog = (root / "CHANGELOG.md").read_text()
        if f"## [{version}]" not in changelog:
            raise ValueError(
                f"Stable release requires an existing changelog entry for [{version}]"
            )
    else:
        title = " ".join(pr["title"].split())
        entry = f"## [{version}]\n\n### Merged\n\n- {title} (#{number})\n\n"
        changelog = (root / "CHANGELOG.md").read_text()
        first_entry = re.search(r"^## \[", changelog, re.MULTILINE)
        if first_entry is None:
            raise ValueError("Changelog must contain a version heading")
        updates["CHANGELOG.md"] = (
            changelog[: first_entry.start()] + entry + changelog[first_entry.start() :]
        )

    for name, content in updates.items():
        (root / name).write_text(content)
    git("add", "--", *updates)
    commit_type = "release" if stable else "chore"
    git(
        "-c",
        "user.name=github-actions[bot]",
        "-c",
        "user.email=41898282+github-actions[bot]@users.noreply.github.com",
        "commit",
        "-m",
        f"{commit_type}: {'release' if stable else 'prerelease'} {tag} from PR #{number}",
    )
    git("tag", tag)
    return tag


def main() -> None:
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    stable = os.environ.get("RELEASE_STABLE", "").lower() in ("1", "true", "yes")
    tag = prepare(Path(__file__).resolve().parents[1], event["pull_request"], stable=stable)
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
        stream.write(f"tag={tag}\n")


if __name__ == "__main__":
    main()
