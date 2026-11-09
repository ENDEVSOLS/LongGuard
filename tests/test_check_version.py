"""Tests for scripts/check_version.py."""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

# Ensure scripts/ is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_version  # noqa: E402


def test_repo_version_consistency():
    """Verify that current repository passes version consistency check."""
    version = check_version.check_version_sync(REPO_ROOT)
    assert version.startswith("0.2.")


def test_version_mismatch_raises(tmp_path: Path):
    """Verify that mismatch between pyproject.toml and __init__.py raises."""
    # Create fake pyproject.toml
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nversion = "0.2.2"\n')

    # Create fake __init__.py
    pkg_dir = tmp_path / "src" / "longguard"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text('__version__ = "0.2.1"\n')

    # Create fake uv.lock
    (tmp_path / "uv.lock").write_text(
        '[[package]]\nname = "longguard"\nversion = "0.2.2"\n'
    )

    with pytest.raises(ValueError, match="version consistency check"):
        check_version.check_version_sync(tmp_path)


def test_tag_mismatch_raises(tmp_path: Path):
    """Verify that mismatch between tag and version raises."""
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.2.2.dev0"\n')
    pkg_dir = tmp_path / "src" / "longguard"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text('__version__ = "0.2.2.dev0"\n')
    (tmp_path / "uv.lock").write_text(
        '[[package]]\nname = "longguard"\nversion = "0.2.2.dev0"\n'
    )

    with pytest.raises(ValueError, match="Tag mismatch"):
        check_version.check_version_sync(tmp_path, tag="v0.2.2")


def test_stable_missing_changelog_raises(tmp_path: Path):
    """Verify that a stable version without a changelog entry raises."""
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.2.2"\n')
    pkg_dir = tmp_path / "src" / "longguard"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text('__version__ = "0.2.2"\n')
    (tmp_path / "uv.lock").write_text(
        '[[package]]\nname = "longguard"\nversion = "0.2.2"\n'
    )
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## [0.2.1] - 2026-11-09\n")

    with pytest.raises(ValueError, match="Missing changelog entry"):
        check_version.check_version_sync(tmp_path)


def test_check_distribution_artifacts_valid(tmp_path: Path):
    """Verify that valid wheel and sdist pass check_distribution_artifacts."""
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()

    wheel_path = dist_dir / "longguard-0.2.2-py3-none-any.whl"
    with zipfile.ZipFile(wheel_path, "w") as zf:
        zf.writestr("longguard/__init__.py", '__version__ = "0.2.2"\n')
        zf.writestr("longguard/py.typed", "")
        zf.writestr(
            "longguard-0.2.2.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: longguard\nVersion: 0.2.2\n",
        )

    sdist_path = dist_dir / "longguard-0.2.2.tar.gz"
    # Create valid sdist tar.gz
    import io

    with tarfile.open(sdist_path, "w:gz") as tf:
        pkg_info = b"Metadata-Version: 2.1\nName: longguard\nVersion: 0.2.2\n"
        ti = tarfile.TarInfo("longguard-0.2.2/PKG-INFO")
        ti.size = len(pkg_info)
        tf.addfile(ti, io.BytesIO(pkg_info))

    check_version.check_distribution_artifacts(tmp_path, dist_dir, "0.2.2")


def test_check_distribution_artifacts_missing_py_typed(tmp_path: Path):
    """Verify that wheel missing py.typed fails distribution check."""
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()

    wheel_path = dist_dir / "longguard-0.2.2-py3-none-any.whl"
    with zipfile.ZipFile(wheel_path, "w") as zf:
        zf.writestr("longguard/__init__.py", '__version__ = "0.2.2"\n')
        zf.writestr(
            "longguard-0.2.2.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: longguard\nVersion: 0.2.2\n",
        )

    sdist_path = dist_dir / "longguard-0.2.2.tar.gz"
    import io

    with tarfile.open(sdist_path, "w:gz") as tf:
        pkg_info = b"Metadata-Version: 2.1\nName: longguard\nVersion: 0.2.2\n"
        ti = tarfile.TarInfo("longguard-0.2.2/PKG-INFO")
        ti.size = len(pkg_info)
        tf.addfile(ti, io.BytesIO(pkg_info))

    with pytest.raises(ValueError, match="missing PEP 561 marker"):
        check_version.check_distribution_artifacts(tmp_path, dist_dir, "0.2.2")
