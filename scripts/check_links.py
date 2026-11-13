"""Automated link and anchor consistency checker for LongGuard documentation."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

# Pattern to extract markdown links: [text](target)
LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
# HTML anchor tags: <a id="..."></a> or <a name="..."></a>
HTML_ANCHOR_PATTERN = re.compile(r'<a\s+(?:id|name)=["\']([^"\']+)["\']', re.IGNORECASE)
# Markdown headers: # Header
HEADER_PATTERN = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)


def slugify_header(title: str) -> str:
    """Convert header text into an anchor slug standard for GitHub / Python-Markdown."""
    # Strip leading markdown hashes if present
    title = re.sub(r"^#{1,6}\s*", "", title)
    # Strip HTML tags
    cleaned = re.sub(r"<[^>]+>", "", title).strip().lower()
    # Remove punctuation
    cleaned = re.sub(r"[^\w\s-]", "", cleaned)
    # Replace whitespace with hyphen and strip leading/trailing hyphens
    return re.sub(r"[-\s]+", "-", cleaned).strip("-")


def collect_anchors(file_path: Path) -> set[str]:
    """Collect all valid anchor IDs and slugified headers in a markdown file."""
    content = file_path.read_text(encoding="utf-8")
    anchors: set[str] = set()

    # Find HTML anchors
    for match in HTML_ANCHOR_PATTERN.finditer(content):
        anchors.add(match.group(1))

    # Find Markdown headers
    for match in HEADER_PATTERN.finditer(content):
        header_text = match.group(1)
        anchors.add(slugify_header(header_text))

    return anchors


def check_file_links(file_path: Path, repo_root: Path) -> list[str]:
    """Audit markdown links in a file, verifying internal targets and anchors."""
    errors: list[str] = []
    content = file_path.read_text(encoding="utf-8")
    local_anchors = collect_anchors(file_path)

    for match in LINK_PATTERN.finditer(content):
        text, target = match.group(1), match.group(2).strip()

        # Ignore mailto and template placeholders
        if target.startswith("mailto:") or "..." in target:
            continue

        parsed = urlparse(target)

        # 1. External URLs (http / https)
        if parsed.scheme in ("http", "https"):
            if not parsed.netloc:
                errors.append(f"{file_path.name}: Malformed URL '{target}'")
            continue

        # 2. Same-page anchor links (#anchor)
        if target.startswith("#"):
            anchor = target[1:]
            if anchor and anchor not in local_anchors:
                if anchor.lower() not in local_anchors:
                    errors.append(
                        f"{file_path.name}: Broken local anchor '#{anchor}' (link text: '{text}')"
                    )
            continue

        # 3. Relative or local file links
        target_path_str = parsed.path
        target_anchor = parsed.fragment

        # Resolve path relative to current file or repo root
        if target_path_str.startswith("/"):
            target_file = repo_root / target_path_str.lstrip("/")
        else:
            target_file = (file_path.parent / target_path_str).resolve()

        if not target_file.exists():
            resolved_rel = (
                target_file.relative_to(repo_root)
                if target_file.is_relative_to(repo_root)
                else target_file
            )
            errors.append(
                f"{file_path.name}: Target file does not exist '{target_path_str}' "
                f"(resolved: {resolved_rel})"
            )
            continue

        if target_anchor and target_file.is_file() and target_file.suffix == ".md":
            target_file_anchors = collect_anchors(target_file)
            if (
                target_anchor not in target_file_anchors
                and target_anchor.lower() not in target_file_anchors
            ):
                errors.append(
                    f"{file_path.name}: Anchor '#{target_anchor}' not found "
                    f"in target '{target_file.name}'"
                )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit documentation and README links.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root")
    args = parser.parse_args()

    repo_root = args.root.resolve()
    docs_dir = repo_root / "docs"

    files_to_check: list[Path] = []
    if (repo_root / "README.md").exists():
        files_to_check.append(repo_root / "README.md")
    if (repo_root / "CONTRIBUTING.md").exists():
        files_to_check.append(repo_root / "CONTRIBUTING.md")

    if docs_dir.exists():
        files_to_check.extend(sorted(docs_dir.rglob("*.md")))

    total_errors: list[str] = []
    checked_count = 0

    for file_path in files_to_check:
        checked_count += 1
        file_errors = check_file_links(file_path, repo_root)
        total_errors.extend(file_errors)

    if total_errors:
        print(f"FAILED: Found {len(total_errors)} broken link(s) across {checked_count} files:\n")
        for err in total_errors:
            print(f"  ✗ {err}")
        return 1

    print(
        f"SUCCESS: Verified links across {checked_count} markdown documentation "
        f"files with zero broken links."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
