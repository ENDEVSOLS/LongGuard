"""Merged PR releases keep source identity and all package versions consistent."""

import importlib.util
import subprocess
import tomllib

import pytest

SPEC = importlib.util.spec_from_file_location(
    "prepare_merge_release", "scripts/prepare_merge_release.py"
)
prepare_merge_release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_merge_release)


@pytest.fixture
def source(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    (tmp_path / "src/longguard").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.2.0.dev0"\n')
    (tmp_path / "src/longguard/__init__.py").write_text('__version__ = "0.2.0.dev0"\n')
    (tmp_path / "uv.lock").write_text('[[package]]\nname = "longguard"\nversion = "0.2.0.dev0"\n')
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## [0.1.0]\n\nPrevious release.\n")
    git("add", ".")
    git("commit", "-qm", "merged PR")
    pr = {
        "number": 42,
        "title": "Fix `guard` and $(touch unwanted-file)",
        "base": {"ref": "main"},
        "merged": True,
        "merge_commit_sha": git("rev-parse", "HEAD"),
    }
    return tmp_path, git, pr


def test_versioned_tag_preserves_source_and_can_be_retried(source):
    root, git, pr = source
    tag = prepare_merge_release.prepare(root, pr)
    assert tag == "v0.2.0.dev42"
    assert git("rev-parse", f"{tag}^") == pr["merge_commit_sha"]
    pyproject_text = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert tomllib.loads(pyproject_text)["project"]["version"] == "0.2.0.dev42"
    init_text = (root / "src/longguard/__init__.py").read_text(encoding="utf-8")
    assert '__version__ = "0.2.0.dev42"' in init_text
    lock_text = (root / "uv.lock").read_text(encoding="utf-8")
    assert tomllib.loads(lock_text)["package"][0]["version"] == "0.2.0.dev42"
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    assert changelog.index("## [0.2.0.dev42]") < changelog.index("## [0.1.0]")
    assert pr["title"] in changelog
    assert not (root / "unwanted-file").exists()
    assert git("status", "--porcelain") == ""
    git("checkout", "--detach", pr["merge_commit_sha"])
    assert prepare_merge_release.prepare(root, pr) == tag
    assert git("rev-parse", "HEAD") == pr["merge_commit_sha"]


@pytest.mark.parametrize(
    "change,message",
    [
        ({"merged": False}, "Only merged"),
        ({"base": {"ref": "other"}}, "Only merged"),
        ({"number": -1}, "positive"),
        ({"merge_commit_sha": "wrong"}, "Checkout"),
    ],
)
def test_rejects_invalid_events_before_modifying_source(source, change, message):
    root, git, pr = source
    with pytest.raises(ValueError, match=message):
        prepare_merge_release.prepare(root, {**pr, **change})
    assert git("status", "--porcelain") == ""
    assert git("tag", "--list") == ""


def test_rejects_tag_for_another_source(source):
    root, git, pr = source
    prepare_merge_release.prepare(root, pr)
    git("commit", "--allow-empty", "-qm", "another merge")
    other = {**pr, "merge_commit_sha": git("rev-parse", "HEAD")}
    # Restore the branch's development metadata as a real later merge would.
    git("checkout", pr["merge_commit_sha"], "--", ".")
    git("commit", "-qm", "restore development version")
    other["merge_commit_sha"] = git("rev-parse", "HEAD")
    with pytest.raises(ValueError, match="different source"):
        prepare_merge_release.prepare(root, other)
