"""Tests for documentation link checker utility."""

from __future__ import annotations

from pathlib import Path

from scripts.check_links import check_file_links, slugify_header


def test_slugify_header() -> None:
    assert slugify_header("Why LongGuard?") == "why-longguard"
    assert slugify_header("## 💡 Why LongGuard?") == "why-longguard"
    assert slugify_header("### 1. LangGraph") == "1-langgraph"
    assert slugify_header("Performance & Overhead") == "performance-overhead"


def test_check_file_links_valid(tmp_path: Path) -> None:
    doc_a = tmp_path / "doc_a.md"
    doc_b = tmp_path / "doc_b.md"

    doc_b.write_text(
        "# Heading Target\n\nTarget content here.\n\n<a id='custom-id'></a>\nAnchor here.",
        encoding="utf-8",
    )
    doc_a.write_text(
        "# Source Document\n\n"
        "[Link to B](doc_b.md)\n"
        "[Link to B header](doc_b.md#heading-target)\n"
        "[Link to B anchor](doc_b.md#custom-id)\n"
        "[External URL](https://github.com/ENDEVSOLS/LongGuard)\n"
        "[Local self anchor](#source-document)\n",
        encoding="utf-8",
    )

    errors = check_file_links(doc_a, tmp_path)
    assert errors == []


def test_check_file_links_broken(tmp_path: Path) -> None:
    doc = tmp_path / "broken.md"
    doc.write_text(
        "# Broken Tests\n\n"
        "[Missing target file](nonexistent_file.md)\n"
        "[Missing local anchor](#missing-anchor)\n",
        encoding="utf-8",
    )

    errors = check_file_links(doc, tmp_path)
    assert len(errors) == 2
    assert any("Target file does not exist" in e for e in errors)
    assert any("Broken local anchor '#missing-anchor'" in e for e in errors)
